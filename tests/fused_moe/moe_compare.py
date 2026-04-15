# SPDX-License-Identifier: Apache-2.0
"""Optimized Triton Fused MoE benchmark and accuracy checker."""

import argparse
import functools
import itertools
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import triton
import triton.language as tl

from vllm.logger import init_logger
from vllm_xpu_kernels import _C, _moe_C

logger = init_logger(__name__)

DEVICE = "xpu"

FUSED_MOE_MNK_FACTORS = [
    (1, 5120, 8192),
    (4, 5120, 8192),
    (16, 5120, 8192),
    (8192, 5120, 8192),
]
NUM_EXPERTS = [16]
TOP_KS = [1]


def format_ms(ms: float) -> str:
    return f"{ms:.3f} ms"


def select_nfused_n(has_bias: bool, use_quant: bool, requested: int) -> int:
    if has_bias or use_quant:
        return 1
    if requested in (1, 2, 4):
        return requested
    return 1


def ref_fused_moe(
    x: torch.Tensor,
    w13: torch.Tensor,
    w13_bias: Optional[torch.Tensor],
    w2: torch.Tensor,
    w2_bias: Optional[torch.Tensor],
    flat_expert_weights: torch.Tensor,
    flat_expert_indices: torch.Tensor,
    num_per_tok: int,
    activation: str,
    num_experts: int,
    ep_rank: int = 0,
    ep_size: int = 1,
) -> torch.Tensor:
    assert activation == "silu", "Only silu is supported in this script."

    expert_start_id = num_experts * ep_rank
    expert_end_id = expert_start_id + num_experts

    expert_cache = torch.zeros_like(x)
    idxs = flat_expert_indices.argsort()
    counts = flat_expert_indices.bincount(minlength=num_experts * ep_size).cpu().numpy()
    tokens_per_expert = counts.cumsum()
    token_idxs = idxs // num_per_tok

    for expert_id, end_idx in enumerate(tokens_per_expert):
        start_idx = 0 if expert_id == 0 else tokens_per_expert[expert_id - 1]
        if (start_idx == end_idx) or (expert_id < expert_start_id) or (expert_id >= expert_end_id):
            continue

        exp_token_idxs = token_idxs[start_idx:end_idx]
        expert_tokens = x[exp_token_idxs]

        expert_w13 = w13[expert_id, :, :]
        w1, w3 = torch.split(expert_w13, expert_w13.shape[0] // 2, dim=0)

        if w13_bias is not None:
            w1_bias, w3_bias = w13_bias[expert_id, :].chunk(2)

        gemm1 = expert_tokens.to(torch.float32) @ w1.T.to(torch.float32)
        if w13_bias is not None:
            gemm1 += w1_bias.to(torch.float32)
        gate = torch.nn.functional.silu(gemm1)

        up = expert_tokens.to(torch.float32) @ w3.T.to(torch.float32)
        if w13_bias is not None:
            up += w3_bias.to(torch.float32)

        expert_mid = gate * up
        expert_out = expert_mid @ w2[expert_id, :, :].T.to(torch.float32)

        if w2_bias is not None:
            expert_out += w2_bias[expert_id, :].to(torch.float32)

        expert_out = expert_out.to(x.dtype)
        expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])

        expert_cache.scatter_reduce_(
            0,
            exp_token_idxs.view(-1, 1).repeat(1, x.shape[-1]),
            expert_out,
            reduce="sum",
        )

    return expert_cache


def make_autotune_configs() -> List[triton.Config]:
    configs: List[triton.Config] = []

    block_m_vals = [16, 32, 64, 128]
    block_n_vals = [16, 32, 64, 128]
    block_k_vals = [16, 32, 64]
    group_m_vals = [1, 4, 8, 16, 32]
    split_k_vals = [1]
    num_warps_vals = [1, 2, 4, 8]

    for bm, bn, bk, gm, sk, nw in itertools.product(
        block_m_vals,
        block_n_vals,
        block_k_vals,
        group_m_vals,
        split_k_vals,
        num_warps_vals,
    ):
        if bm * bn > 128 * 128:
            continue
        if bk > 64:
            continue
        configs.append(
            triton.Config(
                {
                    "BLOCK_SIZE_M": bm,
                    "BLOCK_SIZE_N": bn,
                    "BLOCK_SIZE_K": bk,
                    "GROUP_SIZE_M": gm,
                    "SPLIT_K": sk,
                },
                num_warps=nw,
                num_stages=2,
            )
        )

    if not configs:
        configs = [
            triton.Config(
                {
                    "BLOCK_SIZE_M": 64,
                    "BLOCK_SIZE_N": 32,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 8,
                    "SPLIT_K": 1,
                },
                num_warps=4,
                num_stages=2,
            )
        ]
    return configs


@triton.autotune(
    configs=make_autotune_configs(),
    key=["N", "K", "EM", "num_valid_tokens"],
)
@triton.jit
def fused_moe_kernel_autotuned(
    a_ptr,
    b_ptr,
    c_ptr,
    b_bias_ptr,
    a_scale_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    K,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    stride_bbe,
    stride_bbn,
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    naive_block_assignment: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_channel_quant: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    NFUSED_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N * NFUSED_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    if not naive_block_assignment:
        offs_token_id = pid_m * BLOCK_SIZE_M + offs
        offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    else:
        offs_token = tl.where(offs == 0, pid_m, num_valid_tokens)

    offs_token = offs_token.to(tl.int64)
    token_mask = offs_token < num_valid_tokens
    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    if NFUSED_N >= 2:
        n_tile_base = pid_n * NFUSED_N * BLOCK_SIZE_N
        _bn_range = tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
        offs_bn0 = (n_tile_base + _bn_range) % N
        offs_bn1 = (n_tile_base + BLOCK_SIZE_N + _bn_range) % N
        if NFUSED_N == 4:
            offs_bn2 = (n_tile_base + 2 * BLOCK_SIZE_N + _bn_range) % N
            offs_bn3 = (n_tile_base + 3 * BLOCK_SIZE_N + _bn_range) % N
        offs_bn = offs_bn0
    else:
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N

    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )

    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )

    if NFUSED_N >= 2:
        b_ptrs1 = (
            b_ptr
            + off_experts * stride_be
            + (offs_k[:, None] * stride_bk + offs_bn1[None, :] * stride_bn)
        )
        if NFUSED_N == 4:
            b_ptrs2 = (
                b_ptr
                + off_experts * stride_be
                + (offs_k[:, None] * stride_bk + offs_bn2[None, :] * stride_bn)
            )
            b_ptrs3 = (
                b_ptr
                + off_experts * stride_be
                + (offs_k[:, None] * stride_bk + offs_bn3[None, :] * stride_bn)
            )

    if use_int8_w8a16:
        b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
        b_scale = tl.load(b_scale_ptrs)

    if use_fp8_w8a8 or use_int8_w8a8:
        if group_k > 0 and group_n > 0:
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            offs_bsn = offs_bn // group_n
            b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + offs_bsn * stride_bsn
        elif per_channel_quant:
            b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
            b_scale = tl.load(b_scale_ptrs)
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            a_scale = tl.load(a_scale_ptrs, mask=token_mask, other=0.0)[:, None]
        else:
            a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr + off_experts)

    if HAS_BIAS:
        bias_ptrs = b_bias_ptr + off_experts * stride_bbe + offs_bn * stride_bbn
        bias = tl.load(bias_ptrs, mask=(offs_bn < N), other=0.0)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if NFUSED_N >= 2:
        accumulator1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        if NFUSED_N == 4:
            accumulator2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            accumulator3 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_iter in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < K - k_iter * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=offs_k[:, None] < K - k_iter * BLOCK_SIZE_K,
            other=0.0,
        )

        if NFUSED_N >= 2:
            b1 = tl.load(
                b_ptrs1,
                mask=offs_k[:, None] < K - k_iter * BLOCK_SIZE_K,
                other=0.0,
            )
            if NFUSED_N == 4:
                b2 = tl.load(
                    b_ptrs2,
                    mask=offs_k[:, None] < K - k_iter * BLOCK_SIZE_K,
                    other=0.0,
                )
                b3 = tl.load(
                    b_ptrs3,
                    mask=offs_k[:, None] < K - k_iter * BLOCK_SIZE_K,
                    other=0.0,
                )

        if use_int8_w8a16:
            accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
        elif use_fp8_w8a8 or use_int8_w8a8:
            if group_k > 0 and group_n > 0:
                k_start = k_iter * BLOCK_SIZE_K
                offs_ks = k_start // group_k
                a_scale = tl.load(
                    a_scale_ptrs + offs_ks * stride_ask,
                    mask=token_mask,
                    other=0.0,
                )
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)
                accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
            else:
                if use_fp8_w8a8:
                    accumulator = tl.dot(a, b, acc=accumulator)
                else:
                    accumulator += tl.dot(a, b)
        else:
            accumulator += tl.dot(a, b)
            if NFUSED_N >= 2:
                accumulator1 += tl.dot(a, b1)
                if NFUSED_N == 4:
                    accumulator2 += tl.dot(a, b2)
                    accumulator3 += tl.dot(a, b3)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        if NFUSED_N >= 2:
            b_ptrs1 += BLOCK_SIZE_K * stride_bk
            if NFUSED_N == 4:
                b_ptrs2 += BLOCK_SIZE_K * stride_bk
                b_ptrs3 += BLOCK_SIZE_K * stride_bk

    if use_int8_w8a16:
        accumulator = accumulator * b_scale
    elif (use_fp8_w8a8 or use_int8_w8a8) and not (group_k > 0 and group_n > 0):
        accumulator = accumulator * a_scale * b_scale

    if HAS_BIAS:
        accumulator += bias[None, :]

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator *= moe_weight[:, None]
        if NFUSED_N >= 2:
            accumulator1 *= moe_weight[:, None]
            if NFUSED_N == 4:
                accumulator2 *= moe_weight[:, None]
                accumulator3 *= moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    if NFUSED_N >= 2:
        accumulator1 = accumulator1.to(compute_type)
        if NFUSED_N == 4:
            accumulator2 = accumulator2.to(compute_type)
            accumulator3 = accumulator3.to(compute_type)

    if NFUSED_N == 1:
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)
    else:
        _cn_range = tl.arange(0, BLOCK_SIZE_N)

        offs_cn0 = n_tile_base + _cn_range
        c_ptrs0 = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn0[None, :]
        c_mask0 = token_mask[:, None] & (offs_cn0[None, :] < N)
        tl.store(c_ptrs0, accumulator, mask=c_mask0)

        offs_cn1 = n_tile_base + BLOCK_SIZE_N + _cn_range
        c_ptrs1 = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn1[None, :]
        c_mask1 = token_mask[:, None] & (offs_cn1[None, :] < N)
        tl.store(c_ptrs1, accumulator1, mask=c_mask1)

        if NFUSED_N == 4:
            offs_cn2 = n_tile_base + 2 * BLOCK_SIZE_N + _cn_range
            c_ptrs2 = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn2[None, :]
            c_mask2 = token_mask[:, None] & (offs_cn2[None, :] < N)
            tl.store(c_ptrs2, accumulator2, mask=c_mask2)

            offs_cn3 = n_tile_base + 3 * BLOCK_SIZE_N + _cn_range
            c_ptrs3 = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn3[None, :]
            c_mask3 = token_mask[:, None] & (offs_cn3[None, :] < N)
            tl.store(c_ptrs3, accumulator3, mask=c_mask3)


@triton.jit
def fused_moe_kernel_fixed(
    a_ptr,
    b_ptr,
    c_ptr,
    b_bias_ptr,
    a_scale_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    K,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    stride_bbe,
    stride_bbn,
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    naive_block_assignment: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_channel_quant: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    NFUSED_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N * NFUSED_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    if not naive_block_assignment:
        offs_token_id = pid_m * BLOCK_SIZE_M + offs
        offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    else:
        offs_token = tl.where(offs == 0, pid_m, num_valid_tokens)

    offs_token = offs_token.to(tl.int64)
    token_mask = offs_token < num_valid_tokens
    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    if NFUSED_N >= 2:
        n_tile_base = pid_n * NFUSED_N * BLOCK_SIZE_N
        _bn_range = tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
        offs_bn0 = (n_tile_base + _bn_range) % N
        offs_bn1 = (n_tile_base + BLOCK_SIZE_N + _bn_range) % N
        if NFUSED_N == 4:
            offs_bn2 = (n_tile_base + 2 * BLOCK_SIZE_N + _bn_range) % N
            offs_bn3 = (n_tile_base + 3 * BLOCK_SIZE_N + _bn_range) % N
        offs_bn = offs_bn0
    else:
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N

    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )

    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )

    if NFUSED_N >= 2:
        b_ptrs1 = (
            b_ptr
            + off_experts * stride_be
            + (offs_k[:, None] * stride_bk + offs_bn1[None, :] * stride_bn)
        )
        if NFUSED_N == 4:
            b_ptrs2 = (
                b_ptr
                + off_experts * stride_be
                + (offs_k[:, None] * stride_bk + offs_bn2[None, :] * stride_bn)
            )
            b_ptrs3 = (
                b_ptr
                + off_experts * stride_be
                + (offs_k[:, None] * stride_bk + offs_bn3[None, :] * stride_bn)
            )

    if use_int8_w8a16:
        b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
        b_scale = tl.load(b_scale_ptrs)

    if use_fp8_w8a8 or use_int8_w8a8:
        if group_k > 0 and group_n > 0:
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            offs_bsn = offs_bn // group_n
            b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + offs_bsn * stride_bsn
        elif per_channel_quant:
            b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
            b_scale = tl.load(b_scale_ptrs)
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            a_scale = tl.load(a_scale_ptrs, mask=token_mask, other=0.0)[:, None]
        else:
            a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr + off_experts)

    if HAS_BIAS:
        bias_ptrs = b_bias_ptr + off_experts * stride_bbe + offs_bn * stride_bbn
        bias = tl.load(bias_ptrs, mask=(offs_bn < N), other=0.0)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if NFUSED_N >= 2:
        accumulator1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        if NFUSED_N == 4:
            accumulator2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            accumulator3 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_iter in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < K - k_iter * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=offs_k[:, None] < K - k_iter * BLOCK_SIZE_K,
            other=0.0,
        )

        if NFUSED_N >= 2:
            b1 = tl.load(
                b_ptrs1,
                mask=offs_k[:, None] < K - k_iter * BLOCK_SIZE_K,
                other=0.0,
            )
            if NFUSED_N == 4:
                b2 = tl.load(
                    b_ptrs2,
                    mask=offs_k[:, None] < K - k_iter * BLOCK_SIZE_K,
                    other=0.0,
                )
                b3 = tl.load(
                    b_ptrs3,
                    mask=offs_k[:, None] < K - k_iter * BLOCK_SIZE_K,
                    other=0.0,
                )

        if use_int8_w8a16:
            accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
        elif use_fp8_w8a8 or use_int8_w8a8:
            if group_k > 0 and group_n > 0:
                k_start = k_iter * BLOCK_SIZE_K
                offs_ks = k_start // group_k
                a_scale = tl.load(
                    a_scale_ptrs + offs_ks * stride_ask,
                    mask=token_mask,
                    other=0.0,
                )
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)
                accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
            else:
                if use_fp8_w8a8:
                    accumulator = tl.dot(a, b, acc=accumulator)
                else:
                    accumulator += tl.dot(a, b)
        else:
            accumulator += tl.dot(a, b)
            if NFUSED_N >= 2:
                accumulator1 += tl.dot(a, b1)
                if NFUSED_N == 4:
                    accumulator2 += tl.dot(a, b2)
                    accumulator3 += tl.dot(a, b3)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        if NFUSED_N >= 2:
            b_ptrs1 += BLOCK_SIZE_K * stride_bk
            if NFUSED_N == 4:
                b_ptrs2 += BLOCK_SIZE_K * stride_bk
                b_ptrs3 += BLOCK_SIZE_K * stride_bk

    if use_int8_w8a16:
        accumulator = accumulator * b_scale
    elif (use_fp8_w8a8 or use_int8_w8a8) and not (group_k > 0 and group_n > 0):
        accumulator = accumulator * a_scale * b_scale

    if HAS_BIAS:
        accumulator += bias[None, :]

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator *= moe_weight[:, None]
        if NFUSED_N >= 2:
            accumulator1 *= moe_weight[:, None]
            if NFUSED_N == 4:
                accumulator2 *= moe_weight[:, None]
                accumulator3 *= moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    if NFUSED_N >= 2:
        accumulator1 = accumulator1.to(compute_type)
        if NFUSED_N == 4:
            accumulator2 = accumulator2.to(compute_type)
            accumulator3 = accumulator3.to(compute_type)

    if NFUSED_N == 1:
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)
    else:
        _cn_range = tl.arange(0, BLOCK_SIZE_N)

        offs_cn0 = n_tile_base + _cn_range
        c_ptrs0 = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn0[None, :]
        c_mask0 = token_mask[:, None] & (offs_cn0[None, :] < N)
        tl.store(c_ptrs0, accumulator, mask=c_mask0)

        offs_cn1 = n_tile_base + BLOCK_SIZE_N + _cn_range
        c_ptrs1 = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn1[None, :]
        c_mask1 = token_mask[:, None] & (offs_cn1[None, :] < N)
        tl.store(c_ptrs1, accumulator1, mask=c_mask1)

        if NFUSED_N == 4:
            offs_cn2 = n_tile_base + 2 * BLOCK_SIZE_N + _cn_range
            c_ptrs2 = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn2[None, :]
            c_mask2 = token_mask[:, None] & (offs_cn2[None, :] < N)
            tl.store(c_ptrs2, accumulator2, mask=c_mask2)

            offs_cn3 = n_tile_base + 3 * BLOCK_SIZE_N + _cn_range
            c_ptrs3 = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn3[None, :]
            c_mask3 = token_mask[:, None] & (offs_cn3[None, :] < N)
            tl.store(c_ptrs3, accumulator3, mask=c_mask3)


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sorted_ids = torch.empty(
        (topk_ids.numel() + num_experts * (block_size - 1),),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    expert_ids = torch.empty(
        (topk_ids.numel() + num_experts,),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    sorted_ids.fill_(topk_ids.numel())
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=topk_ids.device)

    torch.ops._moe_C.moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        None,
    )
    return sorted_ids, expert_ids, num_tokens_post_pad


def get_config_file_name(E: int, N: int) -> str:
    device_name = torch.xpu.get_device_name().replace(" ", "_")
    return f"E={E},N={N},device_name={device_name}.json"


@functools.lru_cache
def get_moe_configs(E: int, N: int) -> Optional[Dict[int, Any]]:
    json_file_name = get_config_file_name(E, N)
    config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "configs",
        json_file_name,
    )
    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            logger.info(f"Using configuration from {config_file_path} for MoE layer.")
            return {int(key): val for key, val in json.load(f).items()}
    return None


def default_fixed_config(M: int, E: int) -> Dict[str, Any]:
    config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 32,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
        "SPLIT_K": 1,
        "num_stages": 2,
        "num_warps": 4,
    }
    if M <= E:
        config = {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 16,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 1,
            "SPLIT_K": 1,
            "num_stages": 2,
            "num_warps": 2,
        }
    return config


def invoke_fused_moe_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    mode: str = "autotune",
    config: Optional[Dict[str, Any]] = None,
    nfused_n: int = 4,
    has_bias: bool = False,
) -> None:
    nfused_n = select_nfused_n(has_bias=has_bias, use_quant=False, requested=nfused_n)

    def grid(meta):
        num_pid_m = triton.cdiv(int(num_tokens_post_padded.item()), meta["BLOCK_SIZE_M"])
        num_pid_n = triton.cdiv(B.shape[1], meta["BLOCK_SIZE_N"] * meta["NFUSED_N"])
        return (num_pid_m * num_pid_n,)

    common_kwargs = dict(
        group_n=0,
        group_k=0,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=tl.bfloat16 if A.dtype == torch.bfloat16 else tl.float16,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        per_channel_quant=False,
        naive_block_assignment=(sorted_token_ids is None),
        HAS_BIAS=False,
        NFUSED_N=nfused_n,
    )

    common_args = (
        A,
        B,
        C,
        None,
        None,
        None,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.shape[1],
        B.shape[2],
        int(num_tokens_post_padded.item()),
        topk_ids.numel(),
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )

    if mode == "autotune":
        fused_moe_kernel_autotuned[grid](*common_args, **common_kwargs)
        return

    if mode == "fixed":
        if config is None:
            raise ValueError("config must be provided when mode='fixed'")

        fixed_cfg = config.copy()
        num_warps = fixed_cfg.pop("num_warps", 4)
        num_stages = fixed_cfg.pop("num_stages", 2)

        required = ["BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M", "SPLIT_K"]
        missing = [k for k in required if k not in fixed_cfg]
        if missing:
            raise ValueError(f"fixed config missing keys: {missing}")

        fused_moe_kernel_fixed[grid](
            *common_args,
            **common_kwargs,
            num_warps=num_warps,
            num_stages=num_stages,
            **fixed_cfg,
        )
        return

    raise ValueError(f"Unsupported mode: {mode}")


def optimized_triton_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool = True,
    mode: str = "autotune",
    override_config: Optional[Dict[str, Any]] = None,
    nfused_n: int = 4,
) -> torch.Tensor:
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"
    assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"
    assert gating_output.shape[1] == w1.shape[0], "Number of experts mismatch"
    assert hidden_states.is_contiguous(), "hidden_states must be contiguous"
    assert w1.is_contiguous(), "w1 must be contiguous"
    assert w2.is_contiguous(), "w2 must be contiguous"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]

    M, K = hidden_states.shape
    E, N2, K1 = w1.shape
    assert K == K1
    assert N2 % 2 == 0, "w1.shape[1] must be 2 * intermediate_size"

    intermediate_size = N2 // 2
    assert w2.shape == (E, K, intermediate_size), (
        f"Expected w2 shape {(E, K, intermediate_size)}, got {tuple(w2.shape)}"
    )

    topk_weights, topk_ids = torch.topk(
        gating_output.float(),
        k=topk,
        dim=-1,
        sorted=False,
    )
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    if mode == "fixed":
        if override_config is not None:
            config = override_config
        else:
            configs = get_moe_configs(E, w2.shape[2])
            config = configs[min(configs.keys(), key=lambda x: abs(x - M))] if configs else default_fixed_config(M, E)
        align_block_size = config["BLOCK_SIZE_M"]
    else:
        config = None
        align_block_size = 64 if M > E else 16

    intermediate_cache1 = torch.empty(
        (M, topk, N2),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache2 = torch.empty(
        (M * topk, intermediate_size),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache3 = torch.empty(
        (M, topk, K),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids,
        align_block_size,
        E,
    )

    invoke_fused_moe_kernel(
        hidden_states,
        w1,
        intermediate_cache1,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight=False,
        top_k=topk,
        mode=mode,
        config=config,
        nfused_n=nfused_n,
        has_bias=False,
    )

    _C.silu_and_mul(intermediate_cache2, intermediate_cache1)

    invoke_fused_moe_kernel(
        intermediate_cache2,
        w2,
        intermediate_cache3,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight=True,
        top_k=1,
        mode=mode,
        config=config,
        nfused_n=nfused_n,
        has_bias=False,
    )

    return intermediate_cache3.sum(dim=1)


def benchmark_latency_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        _ = fn()
    torch.xpu.synchronize()

    t0 = time.time()
    for _ in range(iters):
        _ = fn()
    torch.xpu.synchronize()
    t1 = time.time()

    return (t1 - t0) * 1000.0 / iters


def max_abs_and_rel_diff(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float]:
    diff = (a.float() - b.float()).abs()
    max_abs = diff.max().item()
    denom = b.float().abs().clamp_min(1e-6)
    max_rel = (diff / denom).max().item()
    return max_abs, max_rel


def run_single_case(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
    has_bias: bool,
    warmup: int,
    iters: int,
    mode: str,
    nfused_n: int,
):
    torch.manual_seed(7)

    a = torch.randn((m, k), device=DEVICE, dtype=dtype) / 16
    w13 = torch.randn((e, 2 * n, k), device=DEVICE, dtype=dtype) / 16
    w2 = torch.randn((e, k, n), device=DEVICE, dtype=dtype) / 16

    if has_bias:
        w13_bias = torch.randn((e, 2 * n), device=DEVICE, dtype=dtype) / 16
        w2_bias = torch.randn((e, k), device=DEVICE, dtype=dtype) / 16
    else:
        w13_bias = None
        w2_bias = None

    scores = torch.randn((m, e), device=DEVICE, dtype=torch.float32)
    expert_scores, expert_indices = torch.topk(scores, k=topk, dim=-1, sorted=False)

    ref_out = ref_fused_moe(
        a.clone(),
        w13,
        w13_bias,
        w2,
        w2_bias,
        expert_scores.view(-1, 1),
        expert_indices.view(-1),
        topk,
        "silu",
        e,
    )

    if has_bias:
        print(
            f"[SKIP optimized kernel] m={m}, n={n}, k={k}, e={e}, topk={topk}, "
            f"reason=bias path not wired in optimized benchmark"
        )
        return

    fn = lambda: optimized_triton_moe(
        hidden_states=a,
        w1=w13.contiguous(),
        w2=w2.contiguous(),
        gating_output=scores,
        topk=topk,
        renormalize=False,
        mode=mode,
        override_config=None,
        nfused_n=nfused_n,
    )

    out = fn()
    latency_ms = benchmark_latency_ms(fn, warmup=warmup, iters=iters)
    max_abs, max_rel = max_abs_and_rel_diff(out, ref_out)

    if dtype == torch.float16:
        rtol = 1e-2
        atol = 1e-2
    else:
        rtol = 2e-2
        atol = 2e-2

    passed = True
    try:
        torch.testing.assert_close(out, ref_out, rtol=rtol, atol=atol)
    except AssertionError:
        passed = False

    print(
        f"[CASE] mode={mode}, m={m}, n={n}, k={k}, e={e}, topk={topk}, "
        f"dtype={dtype}, NFUSED_N={nfused_n}"
    )
    print(f"  compute latency: {format_ms(latency_ms)}")
    print(f"  accuracy pass  : {passed}")
    print(f"  max abs diff   : {max_abs:.6f}")
    print(f"  max rel diff   : {max_rel:.6f}")
    print("")


def run_bf16_benchmark(
    warmup: int,
    iters: int,
    has_bias: bool,
    mode: str,
    nfused_n: int,
):
    dtype = torch.bfloat16
    for m, n, k in FUSED_MOE_MNK_FACTORS:
        for e in NUM_EXPERTS:
            for topk in TOP_KS:
                run_single_case(
                    m=m,
                    n=n,
                    k=k,
                    e=e,
                    topk=topk,
                    dtype=dtype,
                    has_bias=has_bias,
                    warmup=warmup,
                    iters=iters,
                    mode=mode,
                    nfused_n=nfused_n,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", default=16, type=int)
    parser.add_argument("--hidden", default=4096, type=int)
    parser.add_argument("--intermediate", default=2048, type=int)
    parser.add_argument("--expert", default=64, type=int)
    parser.add_argument("--topk", default=8, type=int)
    parser.add_argument("--warmup", default=10, type=int)
    parser.add_argument("--iters", default=50, type=int)
    parser.add_argument("--loop", default=None, type=int, help="alias of --iters")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--mode", default="autotune", choices=["autotune", "fixed"])
    parser.add_argument("--nfused-n", default=4, type=int, choices=[1, 2, 4])
    parser.add_argument(
        "--run-test-fused-moe-benchmark",
        action="store_true",
        help="Run BF16 benchmark migrated from test_fused_moe.py",
    )
    parser.add_argument(
        "--has-bias",
        action="store_true",
        help="Enable reference bias. Optimized kernel benchmark skips bias cases.",
    )
    args = parser.parse_args()

    if args.loop is not None:
        args.iters = args.loop

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    if args.run_test_fused_moe_benchmark:
        run_bf16_benchmark(
            warmup=args.warmup,
            iters=args.iters,
            has_bias=args.has_bias,
            mode=args.mode,
            nfused_n=args.nfused_n,
        )
    else:
        seq = args.seq
        k = args.hidden
        n = args.intermediate
        expert = args.expert
        topk = args.topk

        print("Begin to initialize weights...")
        a = torch.randn((seq, k), device=DEVICE, dtype=dtype) / 10
        w1 = torch.randn((expert, 2 * n, k), device=DEVICE, dtype=dtype) / 10
        w2 = torch.randn((expert, k, n), device=DEVICE, dtype=dtype) / 10
        gating_output = torch.randn((seq, expert), device=DEVICE, dtype=torch.float32)
        print("Done weights initialization!")

        ref_topk_weights, ref_topk_ids = torch.topk(gating_output, k=topk, dim=-1, sorted=False)
        ref_out = ref_fused_moe(
            a.clone(),
            w1,
            None,
            w2,
            None,
            ref_topk_weights.view(-1, 1),
            ref_topk_ids.view(-1),
            topk,
            "silu",
            expert,
        )

        fn = lambda: optimized_triton_moe(
            hidden_states=a,
            w1=w1.contiguous(),
            w2=w2.contiguous(),
            gating_output=gating_output,
            topk=topk,
            renormalize=False,
            mode=args.mode,
            override_config=None,
            nfused_n=args.nfused_n,
        )

        out = fn()
        latency_ms = benchmark_latency_ms(fn, warmup=args.warmup, iters=args.iters)
        max_abs, max_rel = max_abs_and_rel_diff(out, ref_out)

        print(f"Mode: {args.mode}")
        print(f"Optimized Triton MoE Compute Latency: {format_ms(latency_ms)}")
        print(f"Max abs diff vs reference: {max_abs:.6f}")
        print(f"Max rel diff vs reference: {max_rel:.6f}")
