"""Quick smoke test: load Qwen3-30B-A3B via HF transformers and generate."""

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "/model"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="Hello, I am a language model.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    t0 = time.time()
    print(f"[load] loading tokenizer from {MODEL_DIR} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    print(f"[load] tokenizer ready in {time.time() - t0:.1f}s, vocab={tok.vocab_size}")

    t1 = time.time()
    print("[load] loading model (61GB bf16, device_map=auto across 2x A6000) ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    print(f"[load] model ready in {time.time() - t1:.1f}s")
    mem = torch.cuda.memory_summary(device=None)
    for i in range(torch.cuda.device_count()):
        print(f"[mem] GPU{i}: {torch.cuda.memory_allocated(i)/1e9:.1f} GB allocated")

    inputs = tok(args.prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    t2 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    gen_time = time.time() - t2

    text = tok.decode(out[0], skip_special_tokens=True)
    new_tokens = out.shape[1] - inputs["input_ids"].shape[1]
    print(f"\n[gen] {new_tokens} tokens in {gen_time:.1f}s = {new_tokens/gen_time:.2f} tok/s\n")
    print("=" * 60)
    print(text)
    print("=" * 60)


if __name__ == "__main__":
    main()
