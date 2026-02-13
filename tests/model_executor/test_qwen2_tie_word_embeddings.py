# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.model_executor.models.qwen2 import _resolve_tie_word_embeddings


def test_resolve_tie_word_embeddings_prefers_text_config_value():
    text_config = SimpleNamespace(tie_word_embeddings=False)
    hf_config = SimpleNamespace(tie_word_embeddings=True)

    assert _resolve_tie_word_embeddings(text_config, hf_config) is False


def test_resolve_tie_word_embeddings_falls_back_to_top_level_hf_config():
    text_config = SimpleNamespace()
    hf_config = SimpleNamespace(tie_word_embeddings=True)

    assert _resolve_tie_word_embeddings(text_config, hf_config) is True


def test_resolve_tie_word_embeddings_defaults_to_false_when_missing():
    text_config = SimpleNamespace()
    hf_config = SimpleNamespace()

    assert _resolve_tie_word_embeddings(text_config, hf_config) is False
