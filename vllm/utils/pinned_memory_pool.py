# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch


@dataclass
class _PoolSlot:
    tensor: torch.Tensor
    event: torch.cuda.Event | None


@dataclass
class _RingSlot:
    tensor: torch.Tensor
    event: torch.cuda.Event | None


class PinnedTensorPool:
    def __init__(
        self,
        device: torch.device,
        pool_size: int = 2,
        max_slots: int = 4,
    ) -> None:
        self.device = device
        self.pool_size = pool_size
        self.max_slots = max_slots
        self._pools: dict[torch.dtype, list[_PoolSlot]] = defaultdict(list)
        self._pool_idx: dict[torch.dtype, int] = defaultdict(int)
        self._tensor_map: dict[int, tuple[torch.dtype, int]] = {}

    def acquire(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        *,
        pin_memory: bool,
    ) -> torch.Tensor:
        if not pin_memory:
            return torch.empty(shape, dtype=dtype, device="cpu")

        numel = 1
        for dim in shape:
            numel *= dim
        if numel == 0:
            return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
        key = dtype
        pool = self._pools[key]
        idx = self._pool_idx[key]

        def _alloc_tensor(capacity: int) -> torch.Tensor:
            cap = 1 << (capacity - 1).bit_length()
            return torch.empty((cap,), dtype=dtype, device="cpu", pin_memory=True)

        for _ in range(len(pool)):
            slot = pool[idx]
            if slot.event is None or slot.event.query():
                if slot.tensor.numel() < numel:
                    slot.tensor = _alloc_tensor(numel)
                self._pool_idx[key] = (idx + 1) % len(pool)
                view = slot.tensor[:numel].view(shape)
                self._tensor_map[id(view)] = (key, idx)
                return view
            idx = (idx + 1) % len(pool)

        if len(pool) < self.max_slots:
            tensor = _alloc_tensor(numel)
            pool.append(_PoolSlot(tensor=tensor, event=None))
            idx = len(pool) - 1
            self._pool_idx[key] = (idx + 1) % len(pool)
            view = tensor[:numel].view(shape)
            self._tensor_map[id(view)] = (key, idx)
            return view

        slot = pool[idx]
        if slot.event is not None:
            slot.event.synchronize()
        if slot.tensor.numel() < numel:
            slot.tensor = _alloc_tensor(numel)
        self._pool_idx[key] = (idx + 1) % len(pool)
        view = slot.tensor[:numel].view(shape)
        self._tensor_map[id(view)] = (key, idx)
        return view

    def record_event_if_managed(self, tensor: torch.Tensor) -> None:
        info = self._tensor_map.get(id(tensor))
        if info is None:
            return
        self._tensor_map.pop(id(tensor), None)
        if self.device.type != "cuda":
            return
        key, idx = info
        pool = self._pools[key]
        slot = pool[idx]
        if slot.event is None:
            slot.event = torch.cuda.Event(enable_timing=False)
        slot.event.record(torch.cuda.current_stream(self.device))


class FixedPinnedRingBuffer:
    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        *,
        slots: int = 2,
        pin_memory: bool = True,
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.pin_memory = pin_memory
        self._slots = max(1, slots)
        self._idx = 0
        self._buffers = [
            _RingSlot(
                tensor=torch.empty(
                    shape, dtype=dtype, device="cpu", pin_memory=pin_memory
                ),
                event=None,
            )
            for _ in range(self._slots)
        ]

    def acquire(self, shape: tuple[int, ...]) -> tuple[int, torch.Tensor]:
        if len(shape) != len(self.shape):
            raise ValueError(
                f"Shape rank mismatch: got {shape}, expected {self.shape}."
            )
        for dim, max_dim in zip(shape, self.shape):
            if dim > max_dim:
                raise ValueError(
                    f"Requested shape {shape} exceeds fixed capacity {self.shape}."
                )
        for _ in range(self._slots):
            idx = self._idx
            slot = self._buffers[idx]
            if slot.event is None or slot.event.query():
                self._idx = (idx + 1) % self._slots
                return idx, self._slice(slot.tensor, shape)
            self._idx = (idx + 1) % self._slots

        idx = self._idx
        slot = self._buffers[idx]
        if slot.event is not None:
            slot.event.synchronize()
        self._idx = (idx + 1) % self._slots
        return idx, self._slice(slot.tensor, shape)

    def record(self, idx: int) -> None:
        if not self.pin_memory or self.device.type != "cuda":
            return
        slot = self._buffers[idx]
        if slot.event is None:
            slot.event = torch.cuda.Event(enable_timing=False)
        slot.event.record(torch.cuda.current_stream(self.device))

    @staticmethod
    def _slice(tensor: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
        slices = tuple(slice(0, dim) for dim in shape)
        return tensor[slices]
