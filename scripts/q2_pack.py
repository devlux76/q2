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
import struct
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Magic bytes and version for the binary format.
_HEADER_MAGIC = b"Q2BN"
_FORMAT_VERSION = 1

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
        1  →  fp16 raw   (1-D tensor: bias, layer-norm scale/shift)

    1-D tensors are stored as fp16 to preserve their exact values, since they
    are too small to benefit from Q2 packing and are critical for training
    stability (layer-norm parameters, biases).
    """
    if W.ndim < 2:
        return W.cpu().half().contiguous().numpy().tobytes(), 1

    W_dev = W.to(_DEVICE).float()
    tau   = empirical_tau(W_dev)
    sym   = q2_quantise(W_dev, tau)
    gray  = gray_encode(sym)
    pack  = pack_symbols(gray)
    return pack.cpu().contiguous().numpy().tobytes(), 0


def pack_state_dict(
    state_dict: Dict[str, Tensor],
    out_path: str | Path,
) -> int:
    """Serialise a PyTorch state dict to the Q2 binary format.

    Wire format (all integers big-endian):
        4 B   magic      "Q2BN"
        1 B   version    uint8

        Per tensor (repeated):
            4 B   key_len    uint32
            *     key        UTF-8 bytes
            1 B   ndim       uint8
            4*n   shape      uint32 × ndim
            1 B   dtype_flag uint8  (0 = Q2 packed, 1 = fp16 raw)
            8 B   n_bytes    uint64
            *     data       packed bytes

    Returns the total file size in bytes.
    """
    buf = io.BytesIO()
    buf.write(_HEADER_MAGIC)
    buf.write(struct.pack(">B", _FORMAT_VERSION))

    for key, W in state_dict.items():
        key_b = key.encode()
        buf.write(struct.pack(">I", len(key_b)))
        buf.write(key_b)

        shape = tuple(W.shape)
        buf.write(struct.pack(">B", len(shape)))
        buf.write(struct.pack(f">{len(shape)}I", *shape))

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

    2-D+ tensors are dequantised to {-1.0, -0.5, +0.5, +1.0} unit
    reconstruction points.  This is a valid unit-scale representation;
    callers that need the exact per-row scale must save τ separately.
    """
    raw = Path(in_path).read_bytes()
    if raw[:4] != _HEADER_MAGIC:
        raise ValueError(f"Not a Q2BN file: {in_path}")
    # _ver = raw[4]  # reserved for future version checks
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

        if dtype_flag == 1:
            # fp16 raw
            t = torch.frombuffer(bytearray(data), dtype=torch.float16).to(dtype)
            result[key] = t.reshape(shape).to(device)
        else:
            # Q2 packed: unpack → invert Gray map → dequantise to unit levels
            rows = shape[0]
            cols = int(math.prod(shape[1:]))
            n_packed = math.ceil(cols / 4)
            packed = torch.frombuffer(bytearray(data), dtype=torch.uint8)
            packed = packed.reshape(rows, n_packed)
            gray = unpack_symbols(packed, cols)
            sym  = gray_decode(gray).long()
            # Unit reconstruction: {0,1,2,3} → {-1.0, -0.5, +0.5, +1.0}
            val_map = torch.tensor([-1.0, -0.5, 0.5, 1.0], dtype=dtype)
            W_hat = val_map[sym].reshape(shape)
            result[key] = W_hat.to(device)

    return result


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
