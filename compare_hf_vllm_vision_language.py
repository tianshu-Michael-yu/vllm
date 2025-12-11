# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import dataclasses
import io
import os
import sys
import time
from urllib.request import Request, urlopen

import torch
import torch.multiprocessing as mp
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs

DEFAULT_IMG_URL = "https://www.ilankelman.org/stopsigns/australia.jpg"
DEFAULT_IMG_URL2 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
DEFAULT_PROMPT = "What is in this image?"
DEFAULT_MODEL = "LiquidAI/LFM2-VL-1.6B"


# Disable multiprocessing for debugging
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


def load_image(source: str) -> Image.Image:
    """Load image from URL or local path."""
    try:
        if source.startswith("http://") or source.startswith("https://"):
            req = Request(source, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=30) as r:
                data = r.read()
            img = Image.open(io.BytesIO(data)).convert("RGB")
        else:
            img = Image.open(source).convert("RGB")
        return img
    except Exception as e:
        raise ValueError(f"Failed to load image from {source}: {e}") from e


def run_vllm_offline(
    llm: LLM,
    processor: AutoProcessor,
    image1: Image.Image,
    image2: Image.Image,
    prompt_text: str,
    max_tokens: int = 128,
    temperature: float = 0.0,
    bsz: int = 2,
) -> tuple[list[str], float]:
    # build bsz conversations
    images = []
    conversations = []
    for i in range(bsz):
        if i % 2 == 0:
            images.append(image1)
        else:
            images.append(image2)
        conversations.append(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
        )

    assert processor.chat_template is not None
    texts = [
        processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        for conv in conversations
    ]

    requests = [
        {
            "prompt": text,
            "multi_modal_data": {
                "image": img,
            },
        }
        for text, img in zip(texts, images)
    ]

    sampling = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        # make sure this is a list of ints
        stop_token_ids=(
            [processor.tokenizer.eos_token_id]
            if isinstance(processor.tokenizer.eos_token_id, int)
            else processor.tokenizer.eos_token_id
        ),
    )

    t0 = time.time()
    outputs = llm.generate(requests, sampling_params=sampling)
    dt = time.time() - t0

    # collect texts in the same order
    results = [out.outputs[0].text.strip() for out in outputs]
    return results, dt


def run_hf_transformers(
    model: AutoModelForImageTextToText,
    processor: AutoProcessor,
    image1: Image.Image,
    image2: Image.Image,
    prompt_text: str,
    max_new_tokens: int = 128,
    bsz: int = 2,
) -> tuple[str, float]:
    """Run HF Transformers inference (LLaVA-style)."""

    images = []
    conversations = []
    for i in range(bsz):
        if i % 2 == 0:
            images.append([image1])
        else:
            images.append([image2])
        conversations.append(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
        )

    assert processor.chat_template is not None
    texts = [
        processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        for conversation in conversations
    ]

    # images = [[image] for _ in range(bsz)]

    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(
        model.device
    )

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    dt = time.time() - t0

    # Decode only generated part
    # generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    # decoded = processor.batch_decode(generated_ids, skip_special_tokens=True).strip()
    decoded = processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
    )
    decoded = [d.strip() for d in decoded]
    return decoded, dt


def main():
    parser = argparse.ArgumentParser(
        description="Compare offline vLLM vs HF Transformers VLM inference."
    )
    # parser.add_argument(
    #     "--image", default=DEFAULT_IMG_URL2, help="Image URL or local path"
    # )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="Model ID (shared for both engines)"
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--vllm", action="store_true", help="Run vLLM")
    parser.add_argument("--hf", action="store_true", help="Run HF Transformers")
    parser.add_argument(
        "--device", default=None, help="Override device (e.g., cuda, cpu)"
    )
    args = parser.parse_args()

    if not (args.vllm or args.hf):
        print("Specify --vllm and/or --hf", file=sys.stderr)
        sys.exit(1)

    # Load image
    try:
        img1 = load_image(DEFAULT_IMG_URL)
        img2 = load_image(DEFAULT_IMG_URL2)
    except ValueError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    # print(f"[+] Image: {args.image}")
    print(f"[+] Prompt: {args.prompt}")
    print(f"[+] Model: {args.model}\n")

    processor = None
    if args.hf or (args.vllm):  # Can use for vLLM prompt building
        try:
            processor = AutoProcessor.from_pretrained(args.model)
        except Exception as e:
            print(f"[Processor load error] {e}", file=sys.stderr)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # HF run
    h_text, h_dt = None, None
    if args.hf:
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                args.model,
                device_map="auto" if device == "cuda" else None,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                trust_remote_code=True,
            ).to(device)
            if not processor:
                processor = AutoProcessor.from_pretrained(args.model)
            h_text, h_dt = run_hf_transformers(
                model,
                processor,
                img1,
                img2,
                args.prompt,
                max_new_tokens=args.max_tokens,
            )
        except Exception as e:
            raise e
            print(f"[HF error] {e}", file=sys.stderr)

    # vLLM run
    v_text, v_dt = None, None
    if args.vllm:
        try:
            engine_args = EngineArgs(
                model=args.model,
                max_model_len=4096,
                max_num_seqs=10,
                # mm_processor_kwargs={
                #     "min_pixels": 28 * 28,
                #     "max_pixels": 1280 * 28 * 28,
                #     "fps": 1,
                # },
                limit_mm_per_prompt={"image": 1},
                gpu_memory_utilization=0.8,
                trust_remote_code=True,
                enable_prefix_caching=False,
                block_size=16,
                enforce_eager=True,
            )

            llm = LLM(**dataclasses.asdict(engine_args))
            v_text, v_dt = run_vllm_offline(
                llm,
                processor,
                img1,
                img2,
                args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
        except Exception as e:
            raise e
            print(f"[vLLM error] {e}", file=sys.stderr)

    if args.hf and h_text is not None:
        print("=== HF Transformers ===")
        print(f"Latency: {h_dt:.3f}s")
        print("-" * 100)
        for i, out in enumerate(h_text):
            print(f"Output {i}: {out}")
            print("-" * 100)

    if args.vllm and v_text is not None:
        print("=== vLLM Offline ===")
        print(f"Latency: {v_dt:.3f}s")
        print("-" * 100)
        for i, out in enumerate(v_text):
            print(f"Output {i}: {out}")
            print("-" * 100)


if __name__ == "__main__":
    # 1. python compare_hf_vllm_vision_language.py --model \
    #       HuggingFaceTB/SmolVLM2-500M-Video-Instruct --vllm
    # 2. python compare_hf_vllm_vision_language.py --model \
    #       LiquidAI/LFM2-VL-1.6B --hf
    mp.set_start_method("spawn", force=True)
    main()
