"""Fused MoE kernel."""
import functools
import copy
import json
import os
from typing import Any, Dict, Optional, Tuple

import torch
import triton
import triton.language as tl

from vllm_xpu_kernels import _moe_C, _C
from vllm.logger import init_logger

import time
import argparse

logger = init_logger(__name__)

'''
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8, 16]
        for num_stages in [2, 3, 4, 5, 6]
    ],
    key=["BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M", "SPLIT_K"],
#    restore_value=["Out_ptr"],
)
'''
@triton.jit
def fused_moe_kernel(
    # Pointers to matrices
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
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
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
    stride_bbe,  # bias expert stride
    stride_bbn,  # bias N stride
    # Block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    naive_block_assignment: tl.constexpr,
    # Meta-parameters
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
    # NFUSED_N: number of adjacent N-tiles computed by a single program.
    # When NFUSED_N=2, each program loads A once per K-iteration and uses it
    # to update two separate [BLOCK_SIZE_M, BLOCK_SIZE_N] output tiles.
    # This halves the number of A global-memory reads in the N-direction,
    # which is beneficial on Intel GPU BMG where L2 reuse across programs
    # is limited and A gather traffic (sorted_token_ids indexing) is a
    # significant fraction of memory bandwidth.
    # Tradeoff: doubles B traffic and register pressure per program;
    # only profitable when A traffic dominates.
    # Restricted to the unquantized, bias-free path (NFUSED_N=1 otherwise).
    NFUSED_N: tl.constexpr,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    - naive_block_assignment: A boolean flag indicating whether to use naive
        token wise block assignment. If True, each block corresponds to a
        single token.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    # When NFUSED_N>1 the grid is launched with ceil(N/(BLOCK_SIZE_N*NFUSED_N))
    # programs in the N direction; each program is responsible for NFUSED_N
    # contiguous N-tiles.
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N * NFUSED_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    if not naive_block_assignment:
        offs_token_id = pid_m * BLOCK_SIZE_M + offs
        offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    else:
        offs_token = tl.where(
            offs == 0,
            pid_m,  # first element = pid_m
            num_valid_tokens,  # remaining elements = constant
        )
    # Cast to int64 to prevent overflow in stride*offset products
    # (e.g. stride_cm * offs_token can exceed int32 for large token counts)
    offs_token = offs_token.to(tl.int64)

    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    if NFUSED_N >= 2:
        # NFUSED_N>=2: pid_n indexes N-tile groups; tile 0 starts at
        # pid_n*NFUSED_N*BLOCK_SIZE_N.  A single A load per K-iteration is
        # reused for all B multiplications, cutting A global-memory traffic
        # (beneficial on Intel BMG where gather-A dominates bandwidth).
        n_tile_base = pid_n * NFUSED_N * BLOCK_SIZE_N  # starting column of group
        _bn_range = tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
        offs_bn0 = (n_tile_base + _bn_range) % N
        offs_bn1 = (n_tile_base + BLOCK_SIZE_N + _bn_range) % N
        if NFUSED_N == 4:
            offs_bn2 = (n_tile_base + 2 * BLOCK_SIZE_N + _bn_range) % N
            offs_bn3 = (n_tile_base + 3 * BLOCK_SIZE_N + _bn_range) % N
        # offs_bn aliases offs_bn0 so that quantization/bias code paths that
        # reference offs_bn remain syntactically valid.  Those paths are
        # compile-time dead for NFUSED_N>=2 because all quantization and bias
        # flags are guaranteed False at the call site (see NFUSED_N selection
        # in invoke_fused_moe_triton_kernel).
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
        # Second B-tile pointer for the fused N-tile; advanced in parallel
        # with b_ptrs during the K loop.
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
        b_scale_ptrs = (
            b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
        )
        b_scale = tl.load(b_scale_ptrs)

    if use_fp8_w8a8 or use_int8_w8a8:
        # block-wise
        if group_k > 0 and group_n > 0:
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            offs_bsn = offs_bn // group_n
            b_scale_ptrs = (
                b_scale_ptr + off_experts * stride_bse + offs_bsn * stride_bsn
            )
        # channel-wise
        elif per_channel_quant:
            b_scale_ptrs = (
                b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
            )
            b_scale = tl.load(b_scale_ptrs)
            # Load per-token scale for activations
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            a_scale = tl.load(a_scale_ptrs, mask=token_mask, other=0.0)[:, None]
        # tensor-wise
        else:
            a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr + off_experts)
    if HAS_BIAS:
        # bias shape: [num_experts, N]
        bias_ptrs = b_bias_ptr + off_experts * stride_bbe + offs_bn * stride_bbn
        bias = tl.load(bias_ptrs, mask=(offs_bn < N), other=0.0)
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if NFUSED_N >= 2:
        # Additional accumulators for fused N-tiles.
        accumulator1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        if NFUSED_N == 4:
            accumulator2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            accumulator3 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        if NFUSED_N >= 2:
            # Load additional B-tiles for the same K-block; A is reused.
            b1 = tl.load(
                b_ptrs1, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0
            )
            if NFUSED_N == 4:
                b2 = tl.load(
                    b_ptrs2, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0
                )
                b3 = tl.load(
                    b_ptrs3, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0
                )
        # We accumulate along the K dimension.
        if use_int8_w8a16:
            accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
        elif use_fp8_w8a8 or use_int8_w8a8:
            if group_k > 0 and group_n > 0:
                k_start = k * BLOCK_SIZE_K
                offs_ks = k_start // group_k
                a_scale = tl.load(
                    a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0
                )
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)

                accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
            else:
                if use_fp8_w8a8:
                    # acc used to enable fp8_fast_accum
                    accumulator = tl.dot(a, b, acc=accumulator)
                else:
                    accumulator += tl.dot(a, b)
        else:
            accumulator += tl.dot(a, b)
            if NFUSED_N >= 2:
                # Fused N-tile dots; A is reused from above.
                accumulator1 += tl.dot(a, b1)
                if NFUSED_N == 4:
                    accumulator2 += tl.dot(a, b2)
                    accumulator3 += tl.dot(a, b3)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        if NFUSED_N >= 2:
            b_ptrs1 += BLOCK_SIZE_K * stride_bk
            if NFUSED_N == 4:
                b_ptrs2 += BLOCK_SIZE_K * stride_bk
                b_ptrs3 += BLOCK_SIZE_K * stride_bk

    # Dequantization for supported quantization schemes:
    #   - int8_w8a16
    #   - fp8_w8a8
    #   - int8_w8a8
    # Accumulator and scalings are in float32 to preserve numerical accuracy.
    if use_int8_w8a16:
        accumulator = accumulator * b_scale
    elif (use_fp8_w8a8 or use_int8_w8a8) and not (group_k > 0 and group_n > 0):
        accumulator = accumulator * a_scale * b_scale

    # Bias addition:
    # Bias must be applied after dequantization:
    #   - Since bias is typically not quantized
    #   - Bias should not be scaled by quantization factors
    if HAS_BIAS:
        accumulator += bias[None, :]

    # Router (MoE) weight multiplication:
    # This multiplication MUST be performed in float32 before any precision
    # conversion to ensure numerical stability, which is especially critical
    # on ROCm platforms.
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(
            topk_weights_ptr + offs_token,
            mask=token_mask,
            other=0,
        )
        accumulator *= moe_weight[:, None]
        if NFUSED_N >= 2:
            accumulator1 *= moe_weight[:, None]
            if NFUSED_N == 4:
                accumulator2 *= moe_weight[:, None]
                accumulator3 *= moe_weight[:, None]

    # Final precision conversion:
    # Cast once at the end to the desired compute/output dtype.
    accumulator = accumulator.to(compute_type)
    if NFUSED_N >= 2:
        accumulator1 = accumulator1.to(compute_type)
        if NFUSED_N == 4:
            accumulator2 = accumulator2.to(compute_type)
            accumulator3 = accumulator3.to(compute_type)

    # -----------------------------------------------------------
    # Write back the block(s) of the output.
    # When NFUSED_N>1 each program writes multiple adjacent N-tiles.
    if NFUSED_N == 1:
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)
    else:
        # n_tile_base was computed in the offs_bn setup block above.
        _cn_range = tl.arange(0, BLOCK_SIZE_N)
        offs_cn0 = n_tile_base + _cn_range
        c_ptrs0 = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn0[None, :]
        c_mask0 = token_mask[:, None] & (offs_cn0[None, :] < N)
        tl.store(c_ptrs0, accumulator, mask=c_mask0)
        # Tile 1: n_tile_base+BLOCK_SIZE_N may be >= N for the last program
        # when N is not a multiple of NFUSED_N*BLOCK_SIZE_N.  The B-tile was
        # already loaded with % N wrapping (offs_bn1 above), so the computation
        # is well-defined; the store mask safely discards any out-of-range writes.
        offs_cn1 = n_tile_base + BLOCK_SIZE_N + _cn_range
        c_ptrs1 = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn1[None, :]
        c_mask1 = token_mask[:, None] & (offs_cn1[None, :] < N)
        tl.store(c_ptrs1, accumulator1, mask=c_mask1)
        if NFUSED_N == 4:
            offs_cn2 = n_tile_base + 2 * BLOCK_SIZE_N + _cn_range
            c_ptrs2 = (
                c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn2[None, :]
            )
            c_mask2 = token_mask[:, None] & (offs_cn2[None, :] < N)
            tl.store(c_ptrs2, accumulator2, mask=c_mask2)
            offs_cn3 = n_tile_base + 3 * BLOCK_SIZE_N + _cn_range
            c_ptrs3 = (
                c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn3[None, :]
            )
            c_mask3 = token_mask[:, None] & (offs_cn3[None, :] < N)
            tl.store(c_ptrs3, accumulator3, mask=c_mask3)


def moe_align_block_size(
        topk_ids: torch.Tensor, block_size: int,
        num_experts: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the
        top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according
        to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding,
        ensuring divisibility by block_size.

    This function pads the number of tokens that each expert needs to process
    so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions
    align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]],
    block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts,
        with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids
        [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in
        the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible
        by block_size for proper block matrix operations.
    """
    sorted_ids = torch.empty(
        (topk_ids.numel() + num_experts * (block_size - 1), ),
        dtype=torch.int32,
        device=topk_ids.device)
    expert_ids = torch.empty((topk_ids.numel() + num_experts, ),
                             dtype=torch.int32,
                             device=topk_ids.device)
    sorted_ids.fill_(topk_ids.numel())
    num_tokens_post_pad = torch.empty((1),
                                      dtype=torch.int32,
                                      device=topk_ids.device)
    torch.ops._moe_C.moe_align_block_size(topk_ids, num_experts, block_size, sorted_ids,
                             expert_ids, num_tokens_post_pad, None)
    return sorted_ids, expert_ids, num_tokens_post_pad


def invoke_fused_moe_kernel(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
                            topk_weights: torch.Tensor, topk_ids: torch.Tensor,
                            sorted_token_ids: torch.Tensor,
                            expert_ids: torch.Tensor,
                            num_tokens_post_padded: torch.Tensor,
                            mul_routed_weight: bool, top_k: int,
                            config: Dict[str, Any]) -> None:

    grid = lambda META: (triton.cdiv(sorted_token_ids.shape[0], META[
        'BLOCK_SIZE_M']) * triton.cdiv(B.shape[1], META['BLOCK_SIZE_N']), )
    config = config.copy()
    BLOCK_SIZE_K = config.pop("BLOCK_SIZE_K")
    NFUSED_N = 4

    fused_moe_kernel[grid](
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
        sorted_token_ids.shape[0],
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
        0,
        0,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=tl.bfloat16 if A.dtype == torch.bfloat16 else tl.float16,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        per_channel_quant=False,
        naive_block_assignment=(sorted_token_ids is None),
        HAS_BIAS=False,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        NFUSED_N=NFUSED_N,
        **config,
    )



def get_config_file_name(E: int, N: int) -> str:
    device_name = torch.xpu.get_device_name().replace(" ", "_")
    return f"E={E},N={N},device_name={device_name}.json"


@functools.lru_cache
def get_moe_configs(E: int, N: int) -> Optional[Dict[int, Any]]:
    """
    Return optimized configurations for the fused MoE kernel.

    The return value will be a dictionary that maps an irregular grid of
    batch sizes to configurations of the fused_moe kernel. To evaluate the
    kernel on a given batch size bs, the closest batch size in the grid should
    be picked and the associated configuration chosen to invoke the kernel.
    """

    # First look up if an optimized configuration is available in the configs
    # directory
    json_file_name = get_config_file_name(E, N)

    config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs", json_file_name)
    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            logger.info(
                f"Using configuration from {config_file_path} for MoE layer.")
            # If a configuration has been found, return it
            return {int(key): val for key, val in json.load(f).items()}

    # If no optimized configuration is available, we will use the default
    # configuration
    return None


def fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    loop: int,
    renormalize: bool = True,
    override_config: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """
    This function computes a Mixture of Experts (MoE) layer using two sets of
    weights, w1 and w2, and top-k gating mechanism.

    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
    - w1 (torch.Tensor): The first set of expert weights.
    - w2 (torch.Tensor): The second set of expert weights.
    - gating_output (torch.Tensor): The output of the gating operation
        (before softmax).
    - topk (int): The number of top-k experts to select.
    - renormalize (bool): If True, renormalize the top-k weights to sum to 1.
    - inplace (bool): If True, perform the operation in-place.
        Defaults to False.
    - override_config (Optional[Dict[str, Any]]): Optional override
        for the kernel configuration.

    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    """
    # Check constraints.
    assert hidden_states.shape[0] == gating_output.shape[0], (
        "Number of tokens mismatch")
    assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"
    assert gating_output.shape[1] == w1.shape[0], "Number of experts mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [
        torch.float32, torch.float16, torch.bfloat16
    ]
    M, _ = hidden_states.shape
    E, N, _ = w1.shape

    if True:
        topk_weights = torch.rand(M,
                                  topk,
                                  dtype=torch.float32,
                                  device=hidden_states.device)
        topk_ids = torch.randint(0, expert, (seq, topk), dtype=torch.int32, device="xpu")
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    if override_config:
        config = override_config
    else:
        # First try to load optimal config from the file
        configs = get_moe_configs(E, w2.shape[2])

        if configs:
            # If an optimal configuration map has been found, look up the
            # optimal config
            config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
        else:
            # Else use the default config
            config = {
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 32,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 32,
                "SPLIT_K": 1,
                "num_stages": 1,
                "num_warps": 4
            }

            if M <= E:
                config = {
                    'BLOCK_SIZE_M': 16,
                    'BLOCK_SIZE_N': 16,
                    'BLOCK_SIZE_K': 32,
                    'GROUP_SIZE_M': 1,
                    "SPLIT_K": 1,
                    'num_stages': 1
                }

    intermediate_cache1 = torch.empty((M, topk_ids.shape[1], N),
                                      device=hidden_states.device,
                                      dtype=hidden_states.dtype)
    intermediate_cache2 = torch.empty((M * topk_ids.shape[1], N // 2),
                                      device=hidden_states.device,
                                      dtype=hidden_states.dtype)
    intermediate_cache3 = torch.empty((M, topk_ids.shape[1], w2.shape[1]),
                                      device=hidden_states.device,
                                      dtype=hidden_states.dtype)

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, config['BLOCK_SIZE_M'], E)

    invoke_fused_moe_kernel(hidden_states, w1, intermediate_cache1,
                            topk_weights, topk_ids, sorted_token_ids,
                            expert_ids, num_tokens_post_padded, False,
                            topk_ids.shape[1], config)


    torch.ops._C.silu_and_mul(intermediate_cache2, intermediate_cache1)

    invoke_fused_moe_kernel(intermediate_cache2, w2, intermediate_cache3,
                            topk_weights, topk_ids, sorted_token_ids,
                            expert_ids, num_tokens_post_padded, True, 1,
                            config)

    result = torch.sum(intermediate_cache3.view(*intermediate_cache3.shape),
                       dim=1,
                       out=hidden_states)
    torch.xpu.synchronize()
    print("Done with warm up - Iter #0")

    t0 = time.time()
    for _ in range(loop):
        intermediate_cache1 = torch.empty((M, topk_ids.shape[1], N),
                                          device=hidden_states.device,
                                          dtype=hidden_states.dtype)
        intermediate_cache2 = torch.empty((M * topk_ids.shape[1], N // 2),
                                          device=hidden_states.device,
                                          dtype=hidden_states.dtype)
        intermediate_cache3 = torch.empty((M, topk_ids.shape[1], w2.shape[1]),
                                          device=hidden_states.device,
                                          dtype=hidden_states.dtype)

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, config['BLOCK_SIZE_M'], E)

        invoke_fused_moe_kernel(hidden_states, w1, intermediate_cache1,
                                topk_weights, topk_ids, sorted_token_ids,
                                expert_ids, num_tokens_post_padded, False,
                                topk_ids.shape[1], config)

        torch.ops._C.silu_and_mul(intermediate_cache2, intermediate_cache1)
        
        invoke_fused_moe_kernel(intermediate_cache2, w2, intermediate_cache3,
                                topk_weights, topk_ids, sorted_token_ids,
                                expert_ids, num_tokens_post_padded, True, 1,
                                config)

        result = torch.sum(intermediate_cache3.view(*intermediate_cache3.shape),
                           dim=1,
                           out=hidden_states)
    torch.xpu.synchronize()
    t1 = time.time()
    t = (t1 - t0) * 100.0 / loop
    print("Triton BF16 MoE Compute Latency: {:.3f} ms".format(t))

    return hidden_states


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", default=16, type=int)
    parser.add_argument("--hidden", default=4096, type=int)
    parser.add_argument("--intermediate", default=2048, type=int)
    parser.add_argument("--expert", default=64, type=int)
    parser.add_argument("--topk", default=8, type=int)
    parser.add_argument("--loop", default=50, type=int)
    par = parser.parse_args()

    seq=par.seq
    k=par.hidden
    n=par.intermediate
    expert=par.expert
    topk=par.topk
    loop=par.loop

    print("Begin to initialize weights...")
    # Inputs
    dtype = torch.bfloat16
    a = torch.randn((seq, k), device="xpu", dtype=dtype) / 10
    w1 = torch.randn((expert, 2 * n, k), device="xpu", dtype=dtype) / 10
    w2 = torch.randn((expert, k, n), device="xpu", dtype=dtype) / 10
    gating_output = torch.randn((seq, expert), device="xpu", dtype=dtype)

    print("Done weights initilization!")

    result = fused_moe(
        hidden_states = a,
        w1 = w1,
        w2 = w2,
        gating_output = gating_output,
        topk = topk,
        loop = loop,
    )
