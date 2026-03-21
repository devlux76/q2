(module
  ;; ─────────────────────────────────────────────────────────────────────────────
  ;; Q² — Quaternary Quantisation Kernel (SIMD-accelerated)
  ;; Source:        src/q2.wat
  ;; Specification: DESIGN.md §1.5 – §2.8
  ;;
  ;; Memory layout (8 pages = 512 KB):
  ;;   [0x00000, 0x10000)  page 0 — f32 working buffer (≤ 16 384 dims)
  ;;   [0x10000, 0x20000)  page 1 — reserved / output workspace
  ;;   [0x20000, 0x40000)  pages 2-3 — reserved
  ;;   [0x40000, 0x80000)  pages 4-7 — host input area ($input_ptr must be ≥ 0x40000)
  ;;
  ;; Exports:
  ;;   mem                — shared linear memory (host writes input here, reads output)
  ;;   q2_quantise(...)   — L2-normalise (last token position) + quaternary-quantise → packed Gray bytes
  ;;   q2_key(...)        — run-reduction → 64-bit MSB-aligned transition key
  ;;   q2_lee_distance(…) — SIMD Lee distance via XOR + popcnt on packed Gray vectors
  ;;
  ;; Performance notes (SIMD optimisation):
  ;;   The fp32 dtype path (dtype=0) uses 128-bit SIMD (v128) throughout:
  ;;     • v128.load      — loads 4× f32 in a single instruction
  ;;     • f32x4.mul/add  — accumulates L2 norm across 4 lanes simultaneously
  ;;     • f32x4.splat    — broadcasts norm_inv for parallel normalisation
  ;;     • f32x4.gt       — vectorised threshold comparison (3 compares per 4 dims)
  ;;     • v128.bitselect — branchless symbol classification (no if/else branching)
  ;;     • i32x4.extract_lane — packs 4 Gray codes into one byte
  ;;   Horizontal f32x4 reduction uses i8x16.shuffle (pairwise swap-and-add).
  ;;   The Lee distance function (q2_lee_distance) uses v128.xor + i8x16.popcnt
  ;;   to compute exact cyclic Lee distance in the Z₄ ring without decoding
  ;;   Gray symbols — the fastest hardware-accelerated distance primitive
  ;;   (DESIGN.md §2.6, §2.7 Theorem 2.1: d_H(φ(u),φ(v)) = d_L(u,v)).
  ;;   Non-fp32 dtype paths (fp16, q8, q4, q2) remain scalar since
  ;;   transformers.js always provides fp32 activations on the hot path.
  ;; ─────────────────────────────────────────────────────────────────────────────

  (memory (export "mem") 8)

  ;; Fixed base address for the working buffer (page 0).
  (global $ACCUM_BASE i32 (i32.const 0x00000))

  ;; ─────────────────────────────────────────────────────────────────────────────
  ;; $f16_to_f32 — IEEE 754 half-precision → single-precision conversion
  ;;
  ;; fp16 bit layout (16 bits, stored in the low half of an i32):
  ;;   [15]     sign S
  ;;   [14:10]  biased exponent E  (bias = 15)
  ;;   [9:0]    mantissa M         (10 explicit bits)
  ;;
  ;; fp32 bit layout (32 bits):
  ;;   [31]     sign S
  ;;   [30:23]  biased exponent E  (bias = 127)
  ;;   [22:0]   mantissa M         (23 explicit bits)
  ;;
  ;; Conversion rules:
  ;;   Normal  (E ∈ [1,30]): f32 = sign | ((E+112) << 23) | (M << 13)
  ;;                         (rebias: 127 − 15 = 112)
  ;;   E = 0   (zero/denorm): approximated as ±0.0 (denorms are below quantisation
  ;;                          resolution and contribute negligible energy to the
  ;;                          embedding after L2 normalisation)
  ;;   E = 31  (inf/NaN):     propagated — set f32 exponent to 255, copy mantissa
  ;; ─────────────────────────────────────────────────────────────────────────────
  (func $f16_to_f32 (param $h i32) (result f32)
    (local $sign i32)
    (local $exp  i32)
    (local $mant i32)

    (local.set $sign
      (i32.shl (i32.and (local.get $h) (i32.const 0x8000)) (i32.const 16)))
    (local.set $exp
      (i32.and (i32.shr_u (local.get $h) (i32.const 10)) (i32.const 0x1F)))
    (local.set $mant
      (i32.and (local.get $h) (i32.const 0x3FF)))

    ;; E = 0: zero or denormal — return ±0.0
    (if (i32.eqz (local.get $exp))
      (then (return (f32.reinterpret_i32 (local.get $sign))))
    )

    ;; E = 31: infinity or NaN — propagate with f32 exponent 255
    (if (i32.eq (local.get $exp) (i32.const 31))
      (then
        (return (f32.reinterpret_i32
          (i32.or (local.get $sign)
            (i32.or (i32.const 0x7F800000)
              (i32.shl (local.get $mant) (i32.const 13))))))
      )
    )

    ;; Normal: rebias exponent (add 112) and widen mantissa (shift left 13)
    (f32.reinterpret_i32
      (i32.or (local.get $sign)
        (i32.or
          (i32.shl (i32.add (local.get $exp) (i32.const 112)) (i32.const 23))
          (i32.shl (local.get $mant) (i32.const 13)))))
  )

  ;; ─────────────────────────────────────────────────────────────────────────────
  ;; $read_f32 — read element $idx from input buffer $base with element type $dtype,
  ;; returning the value as f32.
  ;;
  ;; $dtype values and their bit-twiddling:
  ;;
  ;;   0 = fp32   — 4 bytes/element, IEEE 754 single-precision
  ;;               address = $base + $idx × 4
  ;;               load as-is.
  ;;
  ;;   1 = fp16   — 2 bytes/element, IEEE 754 half-precision
  ;;               address = $base + $idx × 2
  ;;               convert via $f16_to_f32 (rebias exponent, widen mantissa).
  ;;
  ;;   2 = q8     — 1 byte/element, signed int8, value ∈ [−128, 127]
  ;;               address = $base + $idx
  ;;               cast directly to f32; L2 normalisation removes scale ambiguity.
  ;;
  ;;   3 = q4     — ½ byte/element, 2 nibbles per byte (unsigned ∈ [0,15])
  ;;               byte address = $base + ($idx >> 1)
  ;;               even $idx → high nibble (byte >> 4)
  ;;               odd  $idx → low  nibble (byte & 0x0F)
  ;;               centred by subtracting 8 → signed ∈ [−8, 7];
  ;;               L2 normalisation removes the fixed ×8 scale factor.
  ;;
  ;;   4 = q2     — ¼ byte/element, 4 symbols per byte, 2 bits each, MSB-first
  ;;               NOTE: this function ($read_f32) is NOT called for dtype=4.
  ;;               q2_quantise detects dtype=4 before entering the accumulation
  ;;               loops and takes a direct byte-copy pass-through path instead
  ;;               (see the q2_quantise function below).  The implementation here
  ;;               is provided only for completeness and defensive correctness in
  ;;               case $read_f32 is ever called with dtype=4 directly.
  ;;               byte address = $base + ($idx >> 2)
  ;;               bit shift    = (3 − ($idx & 3)) × 2   (MSB-first)
  ;;               returns symbol ∈ {0, 1, 2, 3} as f32.
  ;; ─────────────────────────────────────────────────────────────────────────────
  (func $read_f32 (param $base i32) (param $idx i32) (param $dtype i32) (result f32)
    (local $tmp i32)

    (block $b_q2
      (block $b_q4
        (block $b_q8
          (block $b_f16
            (block $b_fp32
              ;; Branch to end of corresponding block; default → q2.
              (br_table $b_fp32 $b_f16 $b_q8 $b_q4 $b_q2 (local.get $dtype))
            )
            ;; dtype = 0: fp32 — 4 bytes per element
            (return
              (f32.load
                (i32.add (local.get $base)
                  (i32.shl (local.get $idx) (i32.const 2)))))
          )
          ;; dtype = 1: fp16 — 2 bytes per element, convert to f32
          (return
            (call $f16_to_f32
              (i32.load16_u
                (i32.add (local.get $base)
                  (i32.shl (local.get $idx) (i32.const 1))))))
        )
        ;; dtype = 2: q8 / int8 — 1 byte per element, signed
        (return
          (f32.convert_i32_s
            (i32.load8_s (i32.add (local.get $base) (local.get $idx)))))
      )

      ;; dtype = 3: q4 — 2 nibbles per byte, unsigned, centred
      ;;   byte address = $base + ($idx >> 1)
      ;;   even idx → high nibble (bits 7:4); odd idx → low nibble (bits 3:0)
      (local.set $tmp
        (i32.load8_u
          (i32.add (local.get $base)
            (i32.shr_u (local.get $idx) (i32.const 1)))))
      (if (i32.and (local.get $idx) (i32.const 1))
        (then
          ;; odd index: low nibble
          (local.set $tmp (i32.and (local.get $tmp) (i32.const 0x0F))))
        (else
          ;; even index: high nibble
          (local.set $tmp (i32.shr_u (local.get $tmp) (i32.const 4)))))
      ;; Centre at zero: subtract 8 → ∈ [−8, 7]
      (return (f32.convert_i32_s (i32.sub (local.get $tmp) (i32.const 8))))
    )

    ;; dtype = 4: q2 — 4 symbols per byte, 2 bits each, MSB-first
    ;;   byte address = $base + ($idx >> 2)
    ;;   right-shift  = (3 − ($idx & 3)) × 2
    (local.set $tmp
      (i32.load8_u
        (i32.add (local.get $base)
          (i32.shr_u (local.get $idx) (i32.const 2)))))
    (f32.convert_i32_u
      (i32.and
        (i32.shr_u (local.get $tmp)
          (i32.shl
            (i32.sub (i32.const 3) (i32.and (local.get $idx) (i32.const 3)))
            (i32.const 1)))
        (i32.const 3)))
  )

  ;; ─────────────────────────────────────────────────────────────────────────────
  ;; q2_quantise — produces the packed Q² Gray-encoded byte vector.
  ;;
  ;; Parameters:
  ;;   $input_ptr  pointer into linear memory where the raw activation tensor begins,
  ;;               laid out as [seq_len × n] in row-major (C) order.
  ;;               For dtype ∈ {0,1,2,3} the element width is 4/2/1/½ bytes.
  ;;               For dtype = 4 the input is n/4 packed Gray-encoded bytes from a
  ;;               prior Q² pass (re-encoding pass; seq_len is ignored).
  ;;   $seq_len    number of token positions (rows); the last position (seq_len − 1)
  ;;               is used; ignored for dtype = 4.
  ;;   $n          native embedding dimension (columns); must be a power of 2, ≤ 16384.
  ;;   $dtype      element dtype: 0=fp32, 1=fp16, 2=q8, 3=q4, 4=q2.
  ;;   $out_ptr    pointer to output buffer; caller must provide ≥ n/4 bytes.
  ;;
  ;; Returns:
  ;;   i32  number of bytes written to $out_ptr (always n/4).
  ;;
  ;; Algorithm (DESIGN.md §1.1, §1.5, §1.7):
  ;;   1. Load:        v[d] = input[(seq_len − 1) × n + d]  (last token position)
  ;;   2. L2-normalise: v[d] /= ‖v‖₂
  ;;   3. Threshold:    τ* = Φ⁻¹(¾) / √n ≈ 0.6745 / √n  (equiprobable 4-cell split)
  ;;   4. Quantise:     sym ∈ {A=0, B=1, C=2, D=3} (see table below)
  ;;   5. Gray-encode:  g = sym ⊕ (sym >> 1)          [DESIGN.md §1.7, Observation]
  ;;   6. Pack:         out[d/4] |= g << (2·(3 − d%4)) (MSB-first within each byte)
  ;;
  ;; Symbol → Z₄ value → Gray encoding (DESIGN.md §1.5, §1.7):
  ;;   A  strong negative  0  00₂   v ≤ −τ*
  ;;   B  weak negative    1  01₂   −τ* < v ≤ 0
  ;;   C  weak positive    2  11₂   0 < v ≤ τ*
  ;;   D  strong positive  3  10₂   v > τ*
  ;;
  ;; Special case — dtype = 4 (q2 input, already packed Gray-encoded bytes):
  ;;   The n/4 input bytes are copied directly from $input_ptr to $out_ptr, and
  ;;   the function returns without performing load/L2-normalise/threshold/
  ;;   quantise/Gray-encode/pack steps.
  ;;
  ;; SIMD fast path (dtype = 0, fp32):
  ;;   When dtype=0 the kernel uses 128-bit SIMD (v128) to process 4 f32
  ;;   dimensions per iteration.  Every loop body operates on v128 registers:
  ;;     Load:       v128.load  (4 f32s from the input tensor in one op)
  ;;     Norm²:      f32x4.mul + f32x4.add  (4-wide FMA accumulation)
  ;;     Normalise:  f32x4.mul with splatted 1/‖v‖  (4-wide broadcast multiply)
  ;;     Quantise:   3× f32x4.gt + v128.not + v128.bitselect  (branchless
  ;;                 symbol classification — no if/else per dimension)
  ;;     Gray + Pack: v128.xor + i32x4.shr_u → i32x4.extract_lane × 4
  ;;                 (4 Gray codes combined into one output byte)
  ;;   Horizontal reduction for the L2 norm uses i8x16.shuffle (pair-wise
  ;;   swap-and-add) to sum the 4 f32 accumulator lanes.
  ;;   Result: the fp32 hot path executes ~4× fewer loop iterations with
  ;;   wider data movement and zero branching in the inner quantisation loop.
  ;; ─────────────────────────────────────────────────────────────────────────────
  (func (export "q2_quantise")
    (param $input_ptr i32)
    (param $seq_len   i32)
    (param $n         i32)
    (param $dtype     i32)
    (param $out_ptr   i32)
    (result i32)

    (local $n_bytes   i32)
    (local $d         i32)
    (local $offset    i32)
    (local $acc_ptr   i32)
    (local $v         f32)
    (local $norm_sq   f32)
    (local $norm_inv  f32)
    (local $tau       f32)
    (local $sym       i32)
    (local $g         i32)
    (local $byte_idx  i32)
    (local $bit_shift i32)
    ;; SIMD locals
    (local $v4         v128)     ;; current 4× f32 vector
    (local $acc4       v128)     ;; SIMD accumulator for norm²
    (local $hi4        v128)     ;; temp for horizontal reduction
    (local $tau4       v128)     ;; splatted threshold
    (local $neg_tau4   v128)     ;; splatted −threshold
    (local $zero4      v128)     ;; splatted 0.0
    (local $mask_a     v128)     ;; v ≤ −τ  (A mask)
    (local $mask_b     v128)     ;; v ≤  0  (A|B mask)
    (local $mask_c     v128)     ;; v ≤  τ  (A|B|C mask)
    (local $sym4       v128)     ;; 4× i32 symbol values
    (local $gray4      v128)     ;; 4× i32 Gray codes
    (local $src_base   i32)      ;; byte offset of last-token row start in input
    (local $packed_byte i32)     ;; assembled output byte

    (local.set $n_bytes (i32.shr_u (local.get $n) (i32.const 2)))

    ;; ── dtype = 4: q2 pass-through ──────────────────────────────────────────
    ;; Input is already n/4 packed Gray-encoded bytes; copy directly to output.
    (if (i32.eq (local.get $dtype) (i32.const 4))
      (then
        (local.set $d (i32.const 0))
        (block $copy_done
          (loop $copy_loop
            (br_if $copy_done (i32.ge_u (local.get $d) (local.get $n_bytes)))
            (i32.store8
              (i32.add (local.get $out_ptr) (local.get $d))
              (i32.load8_u (i32.add (local.get $input_ptr) (local.get $d))))
            (local.set $d (i32.add (local.get $d) (i32.const 1)))
            (br $copy_loop)
          )
        )
        (return (local.get $n_bytes))
      )
    )

    ;; If there are no tokens, nothing to quantise; avoid underflow in
    ;; (seq_len - 1) and return 0 bytes written.
    (if (i32.eqz (local.get $seq_len))
      (then
        (return (i32.const 0))
      )
    )

    ;; ══════════════════════════════════════════════════════════════════════════
    ;; dtype = 0 (fp32): SIMD fast path — process 4 f32 dimensions per iteration
    ;; ══════════════════════════════════════════════════════════════════════════
    (if (i32.eqz (local.get $dtype))
      (then
        ;; Byte offset of last-token row start: input_ptr + (seq_len-1)*n*4
        (local.set $src_base
          (i32.add (local.get $input_ptr)
            (i32.shl
              (i32.mul
                (i32.sub (local.get $seq_len) (i32.const 1))
                (local.get $n))
              (i32.const 2))))

        ;; ── SIMD Step 1: Load last token into working buffer (4 f32s/iter) ──
        (local.set $d (i32.const 0))
        (block $sload_done
          (loop $sload_loop
            (br_if $sload_done (i32.ge_u (local.get $d) (local.get $n)))
            (v128.store
              (i32.add (global.get $ACCUM_BASE) (i32.shl (local.get $d) (i32.const 2)))
              (v128.load
                (i32.add (local.get $src_base) (i32.shl (local.get $d) (i32.const 2)))))
            (local.set $d (i32.add (local.get $d) (i32.const 4)))
            (br $sload_loop)
          )
        )

        ;; ── SIMD Step 2: Compute squared L2 norm (4-wide accumulation) ──────
        (local.set $acc4 (f32x4.splat (f32.const 0.0)))
        (local.set $d (i32.const 0))
        (block $snorm_done
          (loop $snorm_loop
            (br_if $snorm_done (i32.ge_u (local.get $d) (local.get $n)))
            (local.set $v4
              (v128.load
                (i32.add (global.get $ACCUM_BASE) (i32.shl (local.get $d) (i32.const 2)))))
            (local.set $acc4
              (f32x4.add (local.get $acc4)
                (f32x4.mul (local.get $v4) (local.get $v4))))
            (local.set $d (i32.add (local.get $d) (i32.const 4)))
            (br $snorm_loop)
          )
        )

        ;; Horizontal sum: acc4 = [a, b, c, d] → a+b+c+d in lane 0
        ;; Step A: swap lanes [2,3] ↔ [0,1] and add
        (local.set $hi4
          (i8x16.shuffle 8 9 10 11  12 13 14 15  0 1 2 3  4 5 6 7
            (local.get $acc4) (local.get $acc4)))
        (local.set $acc4 (f32x4.add (local.get $acc4) (local.get $hi4)))
        ;; Step B: swap lane 1 ↔ lane 0 and add
        (local.set $hi4
          (i8x16.shuffle 4 5 6 7  0 1 2 3  8 9 10 11  12 13 14 15
            (local.get $acc4) (local.get $acc4)))
        (local.set $acc4 (f32x4.add (local.get $acc4) (local.get $hi4)))
        ;; norm_sq is now in lane 0
        (local.set $norm_sq (f32x4.extract_lane 0 (local.get $acc4)))

        ;; ── SIMD Step 3: L2-normalise (skip if ‖v‖ ≈ 0) ────────────────────
        (if (f32.gt (local.get $norm_sq) (f32.const 1e-16))
          (then
            (local.set $norm_inv
              (f32.div (f32.const 1.0) (f32.sqrt (local.get $norm_sq))))
            (local.set $v4 (f32x4.splat (local.get $norm_inv)))
            (local.set $d (i32.const 0))
            (block $snrm_done
              (loop $snrm_loop
                (br_if $snrm_done (i32.ge_u (local.get $d) (local.get $n)))
                (local.set $acc_ptr
                  (i32.add (global.get $ACCUM_BASE) (i32.shl (local.get $d) (i32.const 2))))
                (v128.store (local.get $acc_ptr)
                  (f32x4.mul (v128.load (local.get $acc_ptr)) (local.get $v4)))
                (local.set $d (i32.add (local.get $d) (i32.const 4)))
                (br $snrm_loop)
              )
            )
          )
        )

        ;; ── SIMD Step 4: Compute threshold τ* = 0.6745 / √n ────────────────
        (local.set $tau
          (f32.div (f32.const 0.6745)
            (f32.sqrt (f32.convert_i32_u (local.get $n)))))
        (local.set $tau4   (f32x4.splat (local.get $tau)))
        (local.set $neg_tau4 (f32x4.splat (f32.neg (local.get $tau))))
        (local.set $zero4  (f32x4.splat (f32.const 0.0)))

        ;; ── SIMD Step 5+6: Quantise → Gray-encode → pack (4 dims → 1 byte) ─
        ;; For every group of 4 f32 values, produce one packed output byte.
        ;; Uses branchless SIMD: 3 vector compares + bitselect (no if/else).
        ;;
        ;; Classification via cascaded ≤ comparisons (≤ = NOT >):
        ;;   mask_a = v ≤ −τ   → where A (sym = 0)
        ;;   mask_b = v ≤  0   → where A or B (sym ≤ 1)
        ;;   mask_c = v ≤  τ   → where A, B, or C (sym ≤ 2)
        ;;
        ;; Symbol selection (start with 3=D, overlay lower values):
        ;;   sym = bitselect(2, 3, mask_c)  →  2 where v ≤ τ, else 3
        ;;   sym = bitselect(1, sym, mask_b) → 1 where v ≤ 0
        ;;   sym = bitselect(0, sym, mask_a) → 0 where v ≤ −τ
        ;;
        ;; Gray-encode: g = sym ⊕ (sym >> 1)  (DESIGN.md §2.7, φ(n) = n ⊕ ⌊n/2⌋)
        ;;
        ;; Pack: extract 4 lanes, shift into MSB-first positions, OR together.
        (local.set $d (i32.const 0))
        (block $sq_done
          (loop $sq_loop
            (br_if $sq_done (i32.ge_u (local.get $d) (local.get $n)))

            ;; Load 4 normalised f32 values
            (local.set $v4
              (v128.load
                (i32.add (global.get $ACCUM_BASE) (i32.shl (local.get $d) (i32.const 2)))))

            ;; Cascaded ≤ masks (≤ is NOT >)
            (local.set $mask_a
              (v128.not (f32x4.gt (local.get $v4) (local.get $neg_tau4))))
            (local.set $mask_b
              (v128.not (f32x4.gt (local.get $v4) (local.get $zero4))))
            (local.set $mask_c
              (v128.not (f32x4.gt (local.get $v4) (local.get $tau4))))

            ;; Branchless symbol selection: D=3 → C=2 → B=1 → A=0
            (local.set $sym4
              (v128.bitselect
                (i32x4.splat (i32.const 2))
                (i32x4.splat (i32.const 3))
                (local.get $mask_c)))
            (local.set $sym4
              (v128.bitselect
                (i32x4.splat (i32.const 1))
                (local.get $sym4)
                (local.get $mask_b)))
            (local.set $sym4
              (v128.bitselect
                (i32x4.splat (i32.const 0))
                (local.get $sym4)
                (local.get $mask_a)))

            ;; Gray-encode: g = sym ⊕ (sym >> 1)
            (local.set $gray4
              (v128.xor (local.get $sym4)
                (i32x4.shr_u (local.get $sym4) (i32.const 1))))

            ;; Pack 4 Gray codes into one byte (MSB-first):
            ;;   byte = g[0]<<6 | g[1]<<4 | g[2]<<2 | g[3]
            (local.set $packed_byte
              (i32.or
                (i32.or
                  (i32.shl (i32x4.extract_lane 0 (local.get $gray4)) (i32.const 6))
                  (i32.shl (i32x4.extract_lane 1 (local.get $gray4)) (i32.const 4)))
                (i32.or
                  (i32.shl (i32x4.extract_lane 2 (local.get $gray4)) (i32.const 2))
                  (i32x4.extract_lane 3 (local.get $gray4)))))

            ;; Write directly — no pre-zeroing needed
            (i32.store8
              (i32.add (local.get $out_ptr) (i32.shr_u (local.get $d) (i32.const 2)))
              (local.get $packed_byte))

            (local.set $d (i32.add (local.get $d) (i32.const 4)))
            (br $sq_loop)
          )
        )

        (return (local.get $n_bytes))
      )
    )

    ;; ══════════════════════════════════════════════════════════════════════════
    ;; Non-fp32 dtypes (1-3): scalar fallback path
    ;; ══════════════════════════════════════════════════════════════════════════

    ;; ── Step 1: Load last token position into working buffer ────────────────
    ;; element index of last token, dimension d: (seq_len − 1) × n + d
    (local.set $d (i32.const 0))
    (block $load_done
      (loop $load_loop
        (br_if $load_done (i32.ge_u (local.get $d) (local.get $n)))
        (local.set $offset
          (i32.add
            (i32.mul (i32.sub (local.get $seq_len) (i32.const 1)) (local.get $n))
            (local.get $d)))
        (local.set $acc_ptr
          (i32.add (global.get $ACCUM_BASE) (i32.shl (local.get $d) (i32.const 2))))
        (f32.store (local.get $acc_ptr)
          (call $read_f32
            (local.get $input_ptr)
            (local.get $offset)
            (local.get $dtype)))
        (local.set $d (i32.add (local.get $d) (i32.const 1)))
        (br $load_loop)
      )
    )

    ;; ── Step 2: Compute squared L2 norm ─────────────────────────────────────
    (local.set $norm_sq (f32.const 0.0))
    (local.set $d (i32.const 0))
    (block $normsq_done
      (loop $normsq_loop
        (br_if $normsq_done (i32.ge_u (local.get $d) (local.get $n)))
        (local.set $v
          (f32.load
            (i32.add (global.get $ACCUM_BASE) (i32.shl (local.get $d) (i32.const 2)))))
        (local.set $norm_sq
          (f32.add (local.get $norm_sq) (f32.mul (local.get $v) (local.get $v))))
        (local.set $d (i32.add (local.get $d) (i32.const 1)))
        (br $normsq_loop)
      )
    )

    ;; ── Step 3: L2-normalise (skip if ‖v‖ ≈ 0) ──────────────────────────────
    (if (f32.gt (local.get $norm_sq) (f32.const 1e-16))
      (then
        ;; norm_inv = 1 / sqrt(norm_sq)
        (local.set $norm_inv
          (f32.div (f32.const 1.0) (f32.sqrt (local.get $norm_sq))))
        (local.set $d (i32.const 0))
        (block $norm_done
          (loop $norm_loop
            (br_if $norm_done (i32.ge_u (local.get $d) (local.get $n)))
            (local.set $acc_ptr
              (i32.add (global.get $ACCUM_BASE) (i32.shl (local.get $d) (i32.const 2))))
            (f32.store (local.get $acc_ptr)
              (f32.mul (f32.load (local.get $acc_ptr)) (local.get $norm_inv)))
            (local.set $d (i32.add (local.get $d) (i32.const 1)))
            (br $norm_loop)
          )
        )
      )
    )

    ;; ── Step 4: Compute threshold τ* = 0.6745 / √n ──────────────────────────
    ;; Φ⁻¹(3/4) ≈ 0.6745; equiprobable 4-cell split for N(0, 1/n) activations.
    ;; (DESIGN.md §1.5)
    (local.set $tau
      (f32.div
        (f32.const 0.6745)
        (f32.sqrt (f32.convert_i32_u (local.get $n)))))

    ;; ── Step 5: Zero output bytes ────────────────────────────────────────────
    (local.set $d (i32.const 0))
    (block $outzero_done
      (loop $outzero_loop
        (br_if $outzero_done (i32.ge_u (local.get $d) (local.get $n_bytes)))
        (i32.store8 (i32.add (local.get $out_ptr) (local.get $d)) (i32.const 0))
        (local.set $d (i32.add (local.get $d) (i32.const 1)))
        (br $outzero_loop)
      )
    )

    ;; ── Step 6: Quantise → Gray-encode → pack ────────────────────────────────
    (local.set $d (i32.const 0))
    (block $quant_done
      (loop $quant_loop
        (br_if $quant_done (i32.ge_u (local.get $d) (local.get $n)))
        (local.set $v
          (f32.load
            (i32.add (global.get $ACCUM_BASE) (i32.shl (local.get $d) (i32.const 2)))))

        ;; Classify into {A=0, B=1, C=2, D=3}
        ;; Default: D (strong positive, v > τ*)
        (local.set $sym (i32.const 3))
        (if (f32.le (local.get $v) (f32.neg (local.get $tau)))
          (then (local.set $sym (i32.const 0)))   ;; A: v ≤ −τ*
          (else
            (if (f32.le (local.get $v) (f32.const 0.0))
              (then (local.set $sym (i32.const 1))) ;; B: −τ* < v ≤ 0
              (else
                (if (f32.le (local.get $v) (local.get $tau))
                  (then (local.set $sym (i32.const 2))) ;; C: 0 < v ≤ τ*
                )
              )
            )
          )
        )

        ;; Gray-encode: g = sym ⊕ (sym >> 1)  (DESIGN.md §1.7, φ(n) = n ⊕ (n >> 1))
        ;; Table: A=0→00, B=1→01, C=2→11, D=3→10
        (local.set $g
          (i32.xor (local.get $sym)
            (i32.shr_u (local.get $sym) (i32.const 1))))

        ;; Pack 4 symbols per byte, MSB-first within each byte:
        ;;   out[d/4] |= g << (2 × (3 − d%4))
        (local.set $byte_idx (i32.shr_u (local.get $d) (i32.const 2)))
        (local.set $bit_shift
          (i32.shl
            (i32.sub (i32.const 3) (i32.and (local.get $d) (i32.const 3)))
            (i32.const 1)))
        (i32.store8
          (i32.add (local.get $out_ptr) (local.get $byte_idx))
          (i32.or
            (i32.load8_u (i32.add (local.get $out_ptr) (local.get $byte_idx)))
            (i32.shl (local.get $g) (local.get $bit_shift))))

        (local.set $d (i32.add (local.get $d) (i32.const 1)))
        (br $quant_loop)
      )
    )

    (local.get $n_bytes)
  )

  ;; ─────────────────────────────────────────────────────────────────────────────
  ;; q2_key — derives the 64-bit MSB-aligned transition key.
  ;;
  ;; Parameters:
  ;;   $packed_ptr  pointer to n/4 packed Gray-encoded bytes (output of q2_quantise)
  ;;   $n           original dimension count
  ;;
  ;; Returns:
  ;;   i64  64-bit key, MSB-aligned: bits 63:62 = first transition, bits 1:0 = 32nd.
  ;;        Symbols beyond position 31 are discarded (DESIGN.md §2.2).
  ;;
  ;; Algorithm (DESIGN.md §2.1 – §2.2):
  ;;   1. Unpack and Gray-decode all n symbols from the packed bytes:
  ;;      g (2-bit Gray) → Z₄ symbol z = (g & 2) | ((g >> 1) ⊕ (g & 1))
  ;;      Verified: g=00→0, g=01→1, g=11→2, g=10→3
  ;;   2. Run-reduction: emit z only when z ≠ previous symbol.
  ;;   3. Pack first 32 transitions into 64-bit key (MSB-aligned):
  ;;      K |= z_i << (62 − 2·i)   for i ∈ [0, min(|transitions|, 32))
  ;; ─────────────────────────────────────────────────────────────────────────────
  (func (export "q2_key")
    (param $packed_ptr i32)
    (param $n          i32)
    (result i64)

    (local $d         i32)
    (local $byte_idx  i32)
    (local $bit_shift i32)
    (local $g         i32)
    (local $z         i32)
    (local $prev      i32)
    (local $key       i64)
    (local $trans     i32)

    ;; prev = 0xFF signals "no previous symbol" (no Z₄ value uses 0xFF)
    (local.set $prev  (i32.const 0xFF))
    (local.set $key   (i64.const 0))
    (local.set $trans (i32.const 0))
    (local.set $d     (i32.const 0))

    (block $dim_done
      (loop $dim_loop
        (br_if $dim_done (i32.ge_u (local.get $d) (local.get $n)))

        ;; Unpack the 2-bit Gray value for dimension $d
        (local.set $byte_idx (i32.shr_u (local.get $d) (i32.const 2)))
        (local.set $bit_shift
          (i32.shl
            (i32.sub (i32.const 3) (i32.and (local.get $d) (i32.const 3)))
            (i32.const 1)))
        (local.set $g
          (i32.and
            (i32.shr_u
              (i32.load8_u (i32.add (local.get $packed_ptr) (local.get $byte_idx)))
              (local.get $bit_shift))
            (i32.const 3)))

        ;; Decode Gray → Z₄:  z = (g & 2) | ((g >> 1) ⊕ (g & 1))
        (local.set $z
          (i32.or
            (i32.and (local.get $g) (i32.const 2))
            (i32.xor
              (i32.shr_u (local.get $g) (i32.const 1))
              (i32.and   (local.get $g) (i32.const 1)))))

        ;; Run-reduction: emit only on transition
        (if (i32.ne (local.get $z) (local.get $prev))
          (then
            (local.set $prev (local.get $z))
            ;; Store if room remains for ≤ 32 transitions
            (if (i32.lt_u (local.get $trans) (i32.const 32))
              (then
                ;; K |= z << (62 − 2 × trans)
                (local.set $key
                  (i64.or (local.get $key)
                    (i64.shl
                      (i64.extend_i32_u (local.get $z))
                      (i64.extend_i32_u
                        (i32.sub (i32.const 62)
                          (i32.shl (local.get $trans) (i32.const 1)))))))
                (local.set $trans (i32.add (local.get $trans) (i32.const 1)))
              )
            )
          )
        )

        (local.set $d (i32.add (local.get $d) (i32.const 1)))
        (br $dim_loop)
      )
    )

    (local.get $key)
  )

  ;; ─────────────────────────────────────────────────────────────────────────────
  ;; q2_lee_distance — SIMD Lee distance between two packed Gray-encoded Q² vectors.
  ;;
  ;; Computes the total Lee distance between two Q² vectors, exploiting the
  ;; isometry established by Theorem 2.1 (Hammons et al., 1994):
  ;;
  ;;   d_H(φ(u), φ(v)) = d_L(u, v)   for all u, v ∈ Z₄ⁿ
  ;;
  ;; where φ is the Gray map (DESIGN.md §2.7) and d_L is the Lee metric on Z₄.
  ;; Because each Q² symbol is Gray-encoded to 2 bits, the Hamming distance
  ;; between the encoded bit-vectors equals the Lee distance on the original
  ;; Z₄ vectors — and Hamming distance is simply popcnt(XOR).
  ;;
  ;; This function uses 128-bit SIMD to XOR 16 packed bytes at a time, count
  ;; set bits per byte with i8x16.popcnt, and horizontally sum via
  ;; i16x8.extadd_pairwise_i8x16_u + i32x4.extadd_pairwise_i16x8_u for a
  ;; fully vectorised distance computation (DESIGN.md §2.6).
  ;;
  ;; Parameters:
  ;;   $a_ptr  pointer to first  packed Gray-encoded vector (n/4 bytes)
  ;;   $b_ptr  pointer to second packed Gray-encoded vector (n/4 bytes)
  ;;   $n      original embedding dimension (n/4 = number of packed bytes)
  ;;
  ;; Returns:
  ;;   i32  total Lee distance (sum of per-dimension Lee distances)
  ;; ─────────────────────────────────────────────────────────────────────────────
  (func (export "q2_lee_distance")
    (param $a_ptr i32)
    (param $b_ptr i32)
    (param $n     i32)
    (result i32)

    (local $n_bytes  i32)
    (local $d        i32)
    (local $total    i32)
    (local $xored    v128)
    (local $popcnt   v128)
    (local $pairs    v128)
    (local $quads    v128)
    (local $hi       v128)

    (local.set $n_bytes (i32.shr_u (local.get $n) (i32.const 2)))
    (local.set $total (i32.const 0))
    (local.set $d (i32.const 0))

    ;; ── SIMD loop: process 16 packed bytes (64 Q² symbols) per iteration ────
    (block $simd_done
      (loop $simd_loop
        ;; Need at least 16 bytes remaining for a SIMD iteration
        (br_if $simd_done
          (i32.gt_u
            (i32.add (local.get $d) (i32.const 16))
            (local.get $n_bytes)))

        ;; XOR corresponding packed bytes: differing bits = Hamming distance bits
        (local.set $xored
          (v128.xor
            (v128.load (i32.add (local.get $a_ptr) (local.get $d)))
            (v128.load (i32.add (local.get $b_ptr) (local.get $d)))))

        ;; Count set bits per byte: each byte's popcount = Lee distance for
        ;; the 4 Q² symbols packed in that byte (Theorem 2.1)
        (local.set $popcnt (i8x16.popcnt (local.get $xored)))

        ;; Horizontal sum of all 16 byte popcounts → single i32 total
        ;; Step 1: pairwise add adjacent u8 → 8× u16
        (local.set $pairs
          (i16x8.extadd_pairwise_i8x16_u (local.get $popcnt)))
        ;; Step 2: pairwise add adjacent u16 → 4× u32
        (local.set $quads
          (i32x4.extadd_pairwise_i16x8_u (local.get $pairs)))
        ;; Step 3: horizontal sum of 4 i32 lanes
        ;; Swap lanes [2,3] ↔ [0,1] and add
        (local.set $hi
          (i8x16.shuffle 8 9 10 11  12 13 14 15  0 1 2 3  4 5 6 7
            (local.get $quads) (local.get $quads)))
        (local.set $quads (i32x4.add (local.get $quads) (local.get $hi)))
        ;; Swap lane 1 ↔ lane 0 and add
        (local.set $hi
          (i8x16.shuffle 4 5 6 7  0 1 2 3  8 9 10 11  12 13 14 15
            (local.get $quads) (local.get $quads)))
        (local.set $quads (i32x4.add (local.get $quads) (local.get $hi)))

        (local.set $total
          (i32.add (local.get $total) (i32x4.extract_lane 0 (local.get $quads))))

        (local.set $d (i32.add (local.get $d) (i32.const 16)))
        (br $simd_loop)
      )
    )

    ;; ── Scalar tail: remaining bytes (< 16) ─────────────────────────────────
    (block $tail_done
      (loop $tail_loop
        (br_if $tail_done (i32.ge_u (local.get $d) (local.get $n_bytes)))

        ;; XOR one byte, count bits with i32.popcnt
        (local.set $total
          (i32.add (local.get $total)
            (i32.popcnt
              (i32.xor
                (i32.load8_u (i32.add (local.get $a_ptr) (local.get $d)))
                (i32.load8_u (i32.add (local.get $b_ptr) (local.get $d)))))))

        (local.set $d (i32.add (local.get $d) (i32.const 1)))
        (br $tail_loop)
      )
    )

    (local.get $total)
  )
)
