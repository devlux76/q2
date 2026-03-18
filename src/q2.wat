(module
  ;; ─────────────────────────────────────────────────────────────────────────────
  ;; Q² — Quaternary Semantic Quantisation Kernel
  ;; Source:        src/q2.wat
  ;; Specification: DESIGN.md §1.5 – §2.2
  ;;
  ;; Memory layout (8 pages = 512 KB):
  ;;   [0x00000, 0x10000)  page 0 — f32 mean-pool accumulator (≤ 16 384 dims)
  ;;   [0x10000, 0x20000)  page 1 — reserved / output workspace
  ;;   [0x20000, 0x40000)  pages 2-3 — reserved
  ;;   [0x40000, 0x80000)  pages 4-7 — host input area ($input_ptr must be ≥ 0x40000)
  ;;
  ;; Exports:
  ;;   mem              — shared linear memory (host writes input here, reads output)
  ;;   q2_quantise(...) — mean-pool + L2-normalise + quaternary-quantise → packed Gray bytes
  ;;   q2_key(...)      — run-reduction → 64-bit MSB-aligned transition key
  ;; ─────────────────────────────────────────────────────────────────────────────

  (memory (export "mem") 8)

  ;; Fixed base address for the mean-pool accumulator buffer (page 0).
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
  ;;               byte address = $base + ($idx >> 2)
  ;;               bit shift    = ($idx & 3) gives position 0..3 within byte;
  ;;                              actual right-shift = (3 − ($idx & 3)) × 2
  ;;               value ∈ {0, 1, 2, 3} — the packed byte already holds Z₄ symbols
  ;;               (prior Q² pass); returned as f32 so the mean-pool arithmetic
  ;;               works uniformly.  Note: because q2 input symbols are ordinal
  ;;               (not magnitude-normalised) the threshold step in q2_quantise
  ;;               is still applied — the symbols are re-quantised relative to
  ;;               their mean, which is a no-op if the symbols are already balanced.
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
  ;;   $seq_len    number of token positions (rows); ignored for dtype = 4.
  ;;   $n          native embedding dimension (columns); must be a power of 2, ≤ 16384.
  ;;   $dtype      element dtype: 0=fp32, 1=fp16, 2=q8, 3=q4, 4=q2.
  ;;   $out_ptr    pointer to output buffer; caller must provide ≥ n/4 bytes.
  ;;
  ;; Returns:
  ;;   i32  number of bytes written to $out_ptr (always n/4).
  ;;
  ;; Algorithm (DESIGN.md §1.5, §1.7):
  ;;   1. Mean-pool:    v[d] = (Σ_s  input[s, d]) / seq_len
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
  ;;   The n/4 input bytes are copied directly to $out_ptr; mean-pooling, L2
  ;;   normalisation, and threshold steps are skipped.
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
    (local $s         i32)
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
    (local $seq_len_f f32)

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

    ;; ── Step 1: Zero the accumulator buffer ─────────────────────────────────
    (local.set $d (i32.const 0))
    (block $zero_done
      (loop $zero_loop
        (br_if $zero_done (i32.ge_u (local.get $d) (local.get $n)))
        (f32.store
          (i32.add (global.get $ACCUM_BASE) (i32.shl (local.get $d) (i32.const 2)))
          (f32.const 0.0))
        (local.set $d (i32.add (local.get $d) (i32.const 1)))
        (br $zero_loop)
      )
    )

    ;; ── Step 2: Accumulate over sequence positions ───────────────────────────
    (local.set $s (i32.const 0))
    (block $seq_done
      (loop $seq_loop
        (br_if $seq_done (i32.ge_u (local.get $s) (local.get $seq_len)))
        (local.set $d (i32.const 0))
        (block $dim_acc_done
          (loop $dim_acc_loop
            (br_if $dim_acc_done (i32.ge_u (local.get $d) (local.get $n)))
            ;; flat tensor index: s × n + d
            (local.set $offset
              (i32.add (i32.mul (local.get $s) (local.get $n)) (local.get $d)))
            (local.set $acc_ptr
              (i32.add (global.get $ACCUM_BASE) (i32.shl (local.get $d) (i32.const 2))))
            (f32.store (local.get $acc_ptr)
              (f32.add
                (f32.load (local.get $acc_ptr))
                (call $read_f32
                  (local.get $input_ptr)
                  (local.get $offset)
                  (local.get $dtype))))
            (local.set $d (i32.add (local.get $d) (i32.const 1)))
            (br $dim_acc_loop)
          )
        )
        (local.set $s (i32.add (local.get $s) (i32.const 1)))
        (br $seq_loop)
      )
    )

    ;; ── Step 3: Divide by seq_len (mean-pool) ────────────────────────────────
    (local.set $seq_len_f (f32.convert_i32_u (local.get $seq_len)))
    (local.set $d (i32.const 0))
    (block $mean_done
      (loop $mean_loop
        (br_if $mean_done (i32.ge_u (local.get $d) (local.get $n)))
        (local.set $acc_ptr
          (i32.add (global.get $ACCUM_BASE) (i32.shl (local.get $d) (i32.const 2))))
        (f32.store (local.get $acc_ptr)
          (f32.div (f32.load (local.get $acc_ptr)) (local.get $seq_len_f)))
        (local.set $d (i32.add (local.get $d) (i32.const 1)))
        (br $mean_loop)
      )
    )

    ;; ── Step 4: Compute squared L2 norm ─────────────────────────────────────
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

    ;; ── Step 5: L2-normalise (skip if ‖v‖ ≈ 0) ──────────────────────────────
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

    ;; ── Step 6: Compute threshold τ* = 0.6745 / √n ──────────────────────────
    ;; Φ⁻¹(3/4) ≈ 0.6745; equiprobable 4-cell split for N(0, 1/n) activations.
    ;; (DESIGN.md §1.5)
    (local.set $tau
      (f32.div
        (f32.const 0.6745)
        (f32.sqrt (f32.convert_i32_u (local.get $n)))))

    ;; ── Step 7: Zero output bytes ────────────────────────────────────────────
    (local.set $d (i32.const 0))
    (block $outzero_done
      (loop $outzero_loop
        (br_if $outzero_done (i32.ge_u (local.get $d) (local.get $n_bytes)))
        (i32.store8 (i32.add (local.get $out_ptr) (local.get $d)) (i32.const 0))
        (local.set $d (i32.add (local.get $d) (i32.const 1)))
        (br $outzero_loop)
      )
    )

    ;; ── Step 8: Quantise → Gray-encode → pack ────────────────────────────────
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
)
