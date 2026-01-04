# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.utils import apply_penalties
from vllm.utils.pinned_memory_pool import FixedPinnedRingBuffer, PinnedTensorPool
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.utils.torch_utils import make_tensor_with_pad


def apply_all_penalties(
    logits: torch.Tensor,
    prompt_token_ids: torch.Tensor,
    presence_penalties: torch.Tensor,
    frequency_penalties: torch.Tensor,
    repetition_penalties: torch.Tensor,
    output_token_ids: list[list[int]],
    output_token_ids_buffer: FixedPinnedRingBuffer | None = None,
    pinned_memory_pool: PinnedTensorPool | None = None,
    pin_memory: bool = False,
) -> torch.Tensor:
    """
    Applies presence, frequency and repetition penalties to the logits.
    """
    _, vocab_size = logits.shape
    output_tokens_t = _convert_to_tensors(
        output_token_ids,
        vocab_size,
        logits.device,
        output_token_ids_buffer=output_token_ids_buffer,
        pinned_memory_pool=pinned_memory_pool,
        pin_memory=pin_memory,
    )

    # In the async scheduling case, rows that won't have penalties applied may contain
    # -1 placeholder token ids. We must replace these with valid token ids so that the
    # scatter done in apply_penalties is valid.
    # NOTE(nick): The penalties implementation is currently quite inefficient and
    # will be reworked anyhow.
    output_tokens_t.masked_fill_(output_tokens_t == -1, vocab_size)

    return apply_penalties(
        logits,
        prompt_token_ids,
        output_tokens_t,
        presence_penalties,
        frequency_penalties,
        repetition_penalties,
    )


def _convert_to_tensors(
    output_token_ids: list[list[int]],
    vocab_size: int,
    device: torch.device,
    *,
    output_token_ids_buffer: FixedPinnedRingBuffer | None,
    pinned_memory_pool: PinnedTensorPool | None,
    pin_memory: bool,
) -> torch.Tensor:
    """
    Convert the different list data structures to tensors.
    """
    if output_token_ids_buffer is not None:
        max_len = max((len(row) for row in output_token_ids), default=0)
        idx, output_tokens_tensor = output_token_ids_buffer.acquire(
            (len(output_token_ids), max_len)
        )
        output_tokens_np = output_tokens_tensor.numpy()
        output_tokens_np.fill(vocab_size)
        for i, row in enumerate(output_token_ids):
            if row:
                output_tokens_np[i, : len(row)] = row
        output_tokens_gpu = output_tokens_tensor.to(device, non_blocking=pin_memory)
        output_token_ids_buffer.record(idx)
        return output_tokens_gpu

    if pin_memory and pinned_memory_pool is not None:
        max_len = max((len(row) for row in output_token_ids), default=0)
        output_tokens_tensor = pinned_memory_pool.acquire(
            (len(output_token_ids), max_len),
            dtype=torch.int64,
            pin_memory=True,
        )
        output_tokens_np = output_tokens_tensor.numpy()
        output_tokens_np.fill(vocab_size)
        for i, row in enumerate(output_token_ids):
            if row:
                output_tokens_np[i, : len(row)] = row
        output_tokens_gpu = output_tokens_tensor.to(device, non_blocking=True)
        pinned_memory_pool.record_event_if_managed(output_tokens_tensor)
        return output_tokens_gpu

    output_tokens_tensor = make_tensor_with_pad(
        output_token_ids,
        # Use the value of vocab_size as a pad since we don't have a
        # token_id of this value.
        pad=vocab_size,
        device="cpu",
        dtype=torch.int64,
        pin_memory=is_pin_memory_available(),
    )
    return output_tokens_tensor.to(device, non_blocking=pin_memory)
