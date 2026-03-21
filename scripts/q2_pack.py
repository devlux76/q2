#!/usr/bin/env python3
"""
q2_pack.py — GPU-accelerated Q² weight packing and unpacking.

Packs PyTorch float32 weight matrices to Q² 2-bit symbols using the Z4
Lee-metric alphabet {A=0, B=1, C=2, D=3}.  Gray-encoded, 4 symbols per byte,
MSB-first — identical to the q2 dtype in src/q2.ts.

All heavy operations run on CUDA when available; falls back to CPU silently.

Public API
----------
    pack_state_dict(state_dict, out_path)  -> int          (artifact bytes)
    unpack_state_dict(in_path, device)     -> dict[str, Tensor]

CLI
---
    python scripts/q2_pack.py model.pt model.q2bin     # pack checkpoint
    python scripts/q2_pack.py --unpack model.q2bin     # inspect packed file
"""
from __future__ import annotations

import argparse
import io
import math
import re
import struct
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Magic bytes and version for the binary format.
_HEADER_MAGIC = b"Q2BN"
_FORMAT_VERSION = 2  # v2 adds per-row τ and alias records

# ── quantisation ──────────────────────────────────────────────────────────────

def empirical_tau(W: Tensor) -> Tensor:
    """Per-row equiprobable threshold τ* from empirical weight statistics.

    Returns the 75th percentile of |W| per row, which equals Φ⁻¹(¾)·σ for
    Gaussian weights (§D-2.5).  The empirical quantile adapts to non-Gaussian
    shapes (e.g. post-ReLU, SwiGLU) without distributional assumptions.

    Args:
        W: (rows, cols) float32 on any device.

    Returns:
        (rows, 1) float32 threshold, same device as W.
    """
    return torch.quantile(W.float().abs(), 0.75, dim=1, keepdim=True).clamp(min=1e-6)


def q2_quantise(W: Tensor, tau: Tensor | None = None) -> Tensor:
    """Quantise float32 weight matrix W to Z4 symbols {A=0, B=1, C=2, D=3}.

    The four equiprobable cells:
        A (0) : w <= -tau   (strong negative)
        B (1) : -tau < w <= 0  (weak negative)
        C (2) : 0 < w <= tau   (weak positive)
        D (3) : w > tau        (strong positive)

    Built with vectorised masks and no Python loops — runs entirely in CUDA
    kernels when W is on a GPU tensor.

    Args:
        W:   (rows, cols) float32.
        tau: (rows, 1) threshold.  Computed via empirical_tau if None.

    Returns:
        (rows, cols) uint8, values in {0, 1, 2, 3}.
    """
    W = W.float()
    if tau is None:
        tau = empirical_tau(W)

    # Build all four masks in parallel; compose sym with integer addition.
    # Start at 0 (A), increment for each boundary crossed.
    # neg_strong → sym stays 0
    sym = (W > -tau).to(torch.uint8)         # 0 if A (w <= -tau), else 1
    sym = sym + (W > 0).to(torch.uint8)      # +1 if past zero  → 1=B or 2=C/D
    sym = sym + (W > tau).to(torch.uint8)    # +1 if past +tau  → 2=C becomes 3=D
    # Result: A=0 B=1 C=2 D=3, all in one pass.
    return sym


def gray_encode(sym: Tensor) -> Tensor:
    """Apply the Gray map φ: Z4 → {0,1,2,3}.

    φ(n) = n XOR (n >> 1): A=0→00, B=1→01, C=2→11, D=3→10.
    Hamming distance on the 2-bit Gray codes equals Lee distance on Z4
    (Theorem 2.1, DESIGN.md §2.7).
    """
    return (sym ^ (sym >> 1)).to(torch.uint8)


def gray_decode(gray: Tensor) -> Tensor:
    """Invert the Gray map (self-inverse for 2-bit codes).

    For 2-bit Gray codes, decoding is the same operation as encoding:
    sym = gray XOR (gray >> 1).
    """
    return (gray ^ (gray >> 1)).to(torch.uint8)


def pack_symbols(gray: Tensor) -> Tensor:
    """Pack 4 Gray-encoded Z4 symbols per byte, MSB-first.

    The packing layout matches src/q2.ts and src/q2.wat:
        byte = (g[4i] << 6) | (g[4i+1] << 4) | (g[4i+2] << 2) | g[4i+3]

    Args:
        gray: (rows, cols) uint8 in {0, 1, 2, 3}.

    Returns:
        (rows, ceil(cols/4)) uint8.  If cols % 4 != 0, the last byte is
        zero-padded on the right.
    """
    rows, cols = gray.shape
    pad = (-cols) % 4
    if pad:
        gray = F.pad(gray, (0, pad), value=0)
    # Reshape to (rows, n_bytes, 4) so each group of 4 symbols is a row.
    g = gray.view(rows, -1, 4).to(torch.int32)
    packed = (g[..., 0] << 6) | (g[..., 1] << 4) | (g[..., 2] << 2) | g[..., 3]
    return packed.to(torch.uint8)


def unpack_symbols(packed: Tensor, n: int) -> Tensor:
    """Unpack bytes to Gray-encoded Z4 symbols.

    Args:
        packed: (rows, ceil(n/4)) uint8.
        n:      number of symbols per row in the original tensor.

    Returns:
        (rows, n) uint8 in {0, 1, 2, 3}.
    """
    p = packed.to(torch.int32)
    s0 = (p >> 6) & 0x3
    s1 = (p >> 4) & 0x3
    s2 = (p >> 2) & 0x3
    s3 =  p       & 0x3
    # Interleave: (rows, n_bytes, 4) → (rows, n_bytes*4) → trim to (rows, n).
    syms = torch.stack([s0, s1, s2, s3], dim=2).view(packed.shape[0], -1)
    return syms[:, :n].to(torch.uint8)


# ── state-dict packing ────────────────────────────────────────────────────────

def pack_tensor(W: Tensor) -> Tuple[bytes, int]:
    """Pack one tensor to Q2 bytes; return (data, dtype_flag).

    dtype_flag meanings:
        0  →  Q2 packed  (2-D or higher weight matrix)
                         data = rows*2 fp16 τ bytes  +  packed symbol bytes
        1  →  fp16 raw   (1-D tensor: bias, layer-norm scale/shift)
        2  →  alias      (handled by pack_state_dict; never returned here)

    Multi-dimensional tensors (ndim > 2) are flattened to (shape[0], prod(shape[1:]))
    before quantisation.  The original shape is stored separately in the header
    so unpack_state_dict can reshape correctly.

    Per-row τ is serialised as fp16 so that unpack_state_dict can dequantise
    weights back to their trained magnitudes, not just unit-scale symbols.
    """
    if W.ndim < 2:
        return W.cpu().half().contiguous().numpy().tobytes(), 1

    # Flatten to 2-D: (rows, cols)
    rows = W.shape[0]
    cols = math.prod(W.shape[1:])
    W_2d = W.reshape(rows, cols)

    W_dev = W_2d.to(_DEVICE).float()
    tau   = empirical_tau(W_dev)          # (rows, 1) float32
    sym   = q2_quantise(W_dev, tau)
    gray  = gray_encode(sym)
    pack  = pack_symbols(gray)            # (rows, ceil(cols/4)) uint8

    # Serialise: fp16 τ (rows × 2 bytes) followed by packed symbols.
    tau_fp16 = tau.squeeze(1).half().cpu().contiguous().numpy().tobytes()
    pack_b   = pack.cpu().contiguous().numpy().tobytes()
    return tau_fp16 + pack_b, 0


def _geode_stratum(key: str) -> Tuple[int, int]:
    """Sort key for Geode-stratum ordering in the binary file.

    Ordering follows the Geode tree traversal (S-1 = S1·G):
        stratum 0 : embedding, emb_norm  (input interface)
        strata 1–4: [GQA, CfC, CfC, CfC] blocks in sequence-order
                    each GQA+CfC group maps to one S1 vertex and its G sub-tree
        stratum 5 : output norm, lm_head (output interface)
        stratum 6 : anything else (buffers etc.)

    Parameters that belong to the same Geode computation unit are adjacent in
    the file, maximising run-length compression (zstd sees long identical-structure
    blocks) and enabling sorted page-through during inference reconstruction.
    """
    if key.startswith(("embed.", "emb_norm.")):
        return (0, 0)

    m = re.match(r"layers\.(\d+)\.", key)
    if m:
        layer_idx = int(m.group(1))
        # Group index: each [GQA+CfC×3] unit = 4 consecutive layer indices.
        group = layer_idx // 4          # 0, 1, 2, 3
        within = layer_idx % 4          # 0=GQA, 1-3=CfC
        # GQA (S1 coarse) sorts before its CfC sub-tree (G refinement).
        return (1 + group, within)

    if key.startswith(("norm.", "lm_head.")):
        return (5, 0)

    return (6, 0)


def pack_state_dict(
    state_dict: Dict[str, Tensor],
    out_path: str | Path,
) -> int:
    """Serialise a PyTorch state dict to the Q2 binary format (v2).

    Wire format (all integers big-endian):
        4 B   magic      "Q2BN"
        1 B   version    uint8 = 2

        Per tensor (repeated, ordered by Geode stratum):
            4 B   key_len    uint32
            *     key        UTF-8 bytes
            1 B   ndim       uint8
            4*n   shape      uint32 × ndim
            1 B   dtype_flag uint8:
                    0 = Q2 packed with per-row τ
                        data = rows*2 fp16 τ + ceil(cols/4)*rows packed bytes
                    1 = fp16 raw (1-D tensors)
                    2 = alias — data is 4-byte key_len + alias_key UTF-8;
                        unpacker must resolve to a previously-loaded tensor.
            8 B   n_bytes    uint64
            *     data       (dtype_flag-specific content above)

    Returns the total file size in bytes.

    Tied weights (embed.weight ≡ lm_head.weight) are deduplicated automatically:
    the first occurrence is serialised in full; subsequent occurrences become
    alias records.  This mirrors the "clustering and collisions are ok" rule
    from the Q² design (§D-2.5): we use the structure to avoid redundancy rather
    than fighting it.
    """
    buf = io.BytesIO()
    buf.write(_HEADER_MAGIC)
    buf.write(struct.pack(">B", _FORMAT_VERSION))

    # Sort entries by Geode stratum so the file layout mirrors the computation
    # tree (§5.5.1: parallel dispatch by tag; §D-4.1: Geode traversal order).
    ordered_keys = sorted(state_dict.keys(), key=_geode_stratum)

    # Track tensors we have already written, keyed by data pointer.
    # Used to emit alias records for tied weights (e.g., embed.weight ≡ lm_head.weight).
    seen_ptrs: Dict[int, str] = {}

    for key in ordered_keys:
        W = state_dict[key]
        key_b = key.encode()
        buf.write(struct.pack(">I", len(key_b)))
        buf.write(key_b)

        shape = tuple(W.shape)
        buf.write(struct.pack(">B", len(shape)))
        buf.write(struct.pack(f">{len(shape)}I", *shape))

        ptr = W.data_ptr()
        if ptr in seen_ptrs:
            # Emit alias record: dtype_flag=2, data = alias_key bytes.
            alias_key_b = seen_ptrs[ptr].encode()
            alias_data  = struct.pack(">I", len(alias_key_b)) + alias_key_b
            buf.write(struct.pack(">BQ", 2, len(alias_data)))
            buf.write(alias_data)
        else:
            seen_ptrs[ptr] = key
            data, dtype_flag = pack_tensor(W)
            buf.write(struct.pack(">BQ", dtype_flag, len(data)))
            buf.write(data)

    payload = buf.getvalue()
    Path(out_path).write_bytes(payload)
    return len(payload)


def unpack_state_dict(
    in_path: str | Path,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Dict[str, Tensor]:
    """Load a Q2BN file back to a float-valued state dict.

    Format v2: per-row τ is stored alongside the packed symbols; dequantised
    values use the saved τ to recover the correct weight magnitudes.
    Format v1 (legacy): unit-scale reconstruction {-1, -0.5, +0.5, +1}.
    Alias records (dtype_flag=2) are resolved to the previously-loaded tensor.
    Multi-dimensional tensors are reshaped back to their original shape.
    """
    raw = Path(in_path).read_bytes()
    if raw[:4] != _HEADER_MAGIC:
        raise ValueError(f"Not a Q2BN file: {in_path}")
    file_version = raw[4]
    pos = 5

    result: Dict[str, Tensor] = {}
    while pos < len(raw):
        (key_len,) = struct.unpack_from(">I", raw, pos)
        pos += 4
        key = raw[pos : pos + key_len].decode()
        pos += key_len

        (ndim,) = struct.unpack_from(">B", raw, pos)
        pos += 1
        shape = struct.unpack_from(f">{ndim}I", raw, pos)
        pos += 4 * ndim

        (dtype_flag,) = struct.unpack_from(">B", raw, pos)
        pos += 1
        (n_bytes,) = struct.unpack_from(">Q", raw, pos)
        pos += 8
        data = raw[pos : pos + n_bytes]
        pos += n_bytes

        if dtype_flag == 2:
            # Alias record: resolve to a previously-loaded tensor.
            (alias_len,) = struct.unpack_from(">I", data, 0)
            alias_key = data[4 : 4 + alias_len].decode()
            result[key] = result[alias_key]
            continue

        if dtype_flag == 1:
            # fp16 raw (biases, norms).
            t = torch.frombuffer(bytearray(data), dtype=torch.float16).to(dtype)
            result[key] = t.reshape(shape).to(device)
            continue

        # dtype_flag == 0: Q2 packed (with per-row τ in v2, without in v1).
        rows = shape[0]
        cols = int(math.prod(shape[1:]))
        n_packed = math.ceil(cols / 4)

        if file_version >= 2:
            # v2: first rows*2 bytes are fp16 τ values.
            tau_bytes = rows * 2
            tau_arr   = torch.frombuffer(bytearray(data[:tau_bytes]), dtype=torch.float16)
            tau_vals  = tau_arr.float().to(device).unsqueeze(1)  # (rows, 1)
            sym_data  = data[tau_bytes:]
        else:
            tau_vals = None
            sym_data = data

        packed = torch.frombuffer(bytearray(sym_data), dtype=torch.uint8)
        packed = packed.reshape(rows, n_packed)
        gray   = unpack_symbols(packed, cols)
        sym    = gray_decode(gray).long()

        if tau_vals is not None:
            # Dequantise using saved τ: {0,1,2,3} → {-1.5,-0.5,+0.5,+1.5}·τ/1.5
            # Reconstruction points at ±0.5τ and ±1.5τ (equiprobable cells §D-2.5).
            val_map = torch.tensor([-1.5, -0.5, 0.5, 1.5], dtype=torch.float32,
                                   device=device)
            W_hat = val_map[sym.to(device)] * (tau_vals / 1.5)
        else:
            # Legacy v1: unit-scale reconstruction.
            val_map = torch.tensor([-1.0, -0.5, 0.5, 1.0], dtype=dtype)
            W_hat = val_map[sym].to(dtype)

        result[key] = W_hat.reshape(shape).to(device)

    return result


# ── LIV cache-line packing (§5.5 of PARAMETER_GOLF.md) ──────────────────────
#
# LIV (Liquid Integrated Vision/Language) symbols use 5-bit quantisation
# (int5, 32 levels).  A 64-bit word can hold:
#
#   12 LIV × 5 bits = 60 bits  +  2-bit tag  +  2 unused bits = 64 bits
#   10 LIV × 5 bits = 50 bits  = two 5×5 binary matrices (codon verifiable)
#
# Exact bit layout (bits 63 … 0, MSB-first):
#   [sym0(5)] [sym1(5)] … [sym11(5)] [tag(2)] [00]
#   bits 63:59  58:54     8:4          3:2     1:0
#
# sym0  → shift = 64 - 5*(0+1) = 59  → bits [63:59]
# sym11 → shift = 64 - 5*(11+1) = 4  → bits [8:4]
# tag   → bits [3:2], values in [0..3] matching the Geode S1 = 4x four levels
# bits [1:0] are unused (zero).
#
# The 2-bit Q² tag distributes 64-bit words across 4 groups for parallel GPU
# warp dispatch by Geode coarse level.


def pack_liv_cacheline(
    symbols: Tensor,
    seq_tags: Tensor | None = None,
) -> Tensor:
    """Pack 5-bit LIV symbols into 64-bit words, 12 per word.

    Packs 12 LIV symbols (values in [0, 31]) per uint64 word with a 2-bit
    Q² sequence tag in bits [3:2].  Bits [1:0] are unused (zero).

    Bit layout (bits 63 → 0):
        sym0[63:59] sym1[58:54] … sym11[8:4] tag[3:2] 00

    The 2-bit tag (4 values = one Q² symbol from the Geode S1 = 4x level)
    allows cache-line-level partitioning across GPU streaming multiprocessors:
    each SM processes one of the 4 tag groups independently.

    Args:
        symbols:  (N,) uint8/int in [0, 31].  Padded to multiple of 12.
        seq_tags: (N//12,) uint8 in [0, 3].  2-bit tag per word.
                  If None, all tags are set to 0.

    Returns:
        (ceil(N/12),) int64 packed words.
    """
    if symbols.numel() % 12 != 0:
        pad = 12 - (symbols.numel() % 12)
        symbols = torch.cat([symbols.flatten(), symbols.new_zeros(pad)])
    n_words = symbols.numel() // 12
    s = symbols.view(n_words, 12).to(torch.int64) & 0x1F  # 5-bit clamp

    # sym0 → shift=59 (bits [63:59]), sym11 → shift=4 (bits [8:4]).
    word = torch.zeros(n_words, dtype=torch.int64, device=symbols.device)
    for i in range(12):
        shift = 64 - 5 * (i + 1)   # sym0→59, sym1→54, …, sym11→4
        word |= s[:, i] << shift

    # 2-bit tag in bits [3:2]; bits [1:0] remain zero.
    if seq_tags is not None:
        tag = seq_tags.view(n_words).to(torch.int64) & 0x3
        word |= tag << 2
    return word


def unpack_liv_cacheline(packed: Tensor, n: int) -> Tuple[Tensor, Tensor]:
    """Unpack 64-bit words to 5-bit LIV symbols and 2-bit Q² tags.

    Args:
        packed: (N_words,) int64.
        n:      total number of symbols to return (≤ N_words × 12).

    Returns:
        symbols:  (n,) uint8 in [0, 31].
        seq_tags: (N_words,) uint8 in [0, 3].
    """
    n_words = packed.shape[0]
    out = torch.zeros(n_words * 12, dtype=torch.uint8, device=packed.device)
    for i in range(12):
        shift = 64 - 5 * (i + 1)   # matches pack_liv_cacheline
        out[i::12] = ((packed >> shift) & 0x1F).to(torch.uint8)
    seq_tags = ((packed >> 2) & 0x3).to(torch.uint8)
    return out[:n], seq_tags


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack / inspect a Q2 weight binary.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input",  help="Input .pt checkpoint or .q2bin file")
    parser.add_argument("output", nargs="?", help="Output .q2bin path (pack mode)")
    parser.add_argument("--unpack", action="store_true", help="Inspect a .q2bin file")
    args = parser.parse_args()

    if args.unpack or args.input.endswith(".q2bin"):
        sd = unpack_state_dict(args.input)
        total = sum(t.numel() for t in sd.values())
        print(f"Loaded {len(sd)} tensors, {total:,} total elements")
        for k, v in sd.items():
            print(f"  {k:<50s} {str(tuple(v.shape)):<25s} {v.dtype}")
        return

    if not args.output:
        parser.error("Provide output path (or --unpack to inspect)")

    sd = torch.load(args.input, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]

    n_bytes = pack_state_dict(sd, args.output)
    print(f"Packed {len(sd)} tensors → {n_bytes:,} bytes ({n_bytes / 1e6:.3f} MB)")
    print(f"Device used: {_DEVICE}")


if __name__ == "__main__":
    main()
