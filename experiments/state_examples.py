"""Model-written examples of each state's text in the per-state caches' opening (trial).

    H2O_LIGHTNING_PATH=<checkout> uv run python -I experiments/state_examples.py write     # 2 passages per state
    H2O_LIGHTNING_PATH=<checkout> uv run python -I experiments/wording_screen.py run --spec per-state-examples

A one-line state label at the top barely moves the model's text prediction (docs/LOG.md, 2026-11-26 and 27). Here
each state's cache opening also carries two short passages (up to 120 tokens each) that the post-trained Qwen3.5-4B
writes on request from the state's description ("Write a short passage of text that is about <description>. Write
only the passage."), placed as "Text in state <name>: ..."
before "The end of the following text is in state: <name>". Self-generated in-context examples are a known zero-shot
technique (Kim et al. 2022, not opened here); this opening is the project's own form. Seed 0, temperature 1.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import joined_documents as jd  # noqa: E402
import state_vocab as sv  # noqa: E402
import wording_screen as ws  # noqa: E402

OUT = sv.OUT / "examples.json"


def write():
    import torch
    from transformers import AutoModelForCausalLM
    tok = jd.tokenizer()
    post = next(Path.home().glob(".cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/*/")).as_posix()
    model = AutoModelForCausalLM.from_pretrained(post, dtype=torch.bfloat16, device_map="cuda").eval()
    torch.manual_seed(0)
    out = {}
    for s, (states, _) in ws.screen_docs(tok).items():
        out[s] = {}
        for name, desc in states.items():
            msg = [{"role": "user", "content": f"Write a short passage of text that is about {desc}. Write only the passage."}]
            text = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            prompt = tok(text, add_special_tokens=False)["input_ids"]
            with torch.no_grad():
                gen = model.generate(torch.tensor([prompt] * 2, device="cuda"), do_sample=True, temperature=1.0,
                                     max_new_tokens=120, pad_token_id=tok.pad_token_id or tok.eos_token_id)
            out[s][name] = "\n\n".join(tok.decode(g[len(prompt):], skip_special_tokens=True).strip() for g in gen)
    OUT.write_text(json.dumps(out, indent=1))
    print("written", {s: len(v) for s, v in out.items()})


if __name__ == "__main__":
    {"write": write}[sys.argv[1]]()
