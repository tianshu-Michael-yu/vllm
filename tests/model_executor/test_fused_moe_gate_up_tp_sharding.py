import torch


def _make_loaded_gate_up(num_experts: int, inter: int, hidden: int) -> torch.Tensor:
    # Shape: [E, 2*inter, hidden]. Fill deterministically to validate slicing.
    t = torch.arange(num_experts * 2 * inter * hidden, dtype=torch.float32)
    return t.reshape(num_experts, 2 * inter, hidden)


def test_load_combined_gate_up_proj_tp2_shards_gate_and_up_separately():
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

    num_experts, inter, hidden = 1, 4, 3
    loaded = _make_loaded_gate_up(num_experts, inter, hidden)

    gate = loaded[:, :inter, :]
    up = loaded[:, inter:, :]
    shard = inter // 2

    for tp_rank in (0, 1):
        op = FusedMoE.__new__(FusedMoE)
        # tp_size/tp_rank are properties backed by moe_parallel_config.
        op.moe_parallel_config = type("_Cfg", (), {"tp_size": 2, "tp_rank": tp_rank})()

        param = torch.nn.Parameter(torch.empty((num_experts, 2 * shard, hidden)))
        ok = FusedMoE.load_combined_gate_up_proj(op, param, loaded)
        assert ok is True

        start = tp_rank * shard
        end = start + shard
        expected = torch.cat((gate[:, start:end, :], up[:, start:end, :]), dim=1)
        assert torch.equal(param.data, expected)


def test_load_combined_gate_up_proj_rejects_odd_intermediate_dim():
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

    op = FusedMoE.__new__(FusedMoE)
    op.moe_parallel_config = type("_Cfg", (), {"tp_size": 2, "tp_rank": 0})()

    loaded = torch.zeros((1, 7, 3), dtype=torch.float32)  # 7 is odd => invalid
    param = torch.nn.Parameter(torch.empty((1, 4, 3)))
    ok = FusedMoE.load_combined_gate_up_proj(op, param, loaded)
    assert ok is False

