#!/usr/bin/env python3
"""
Stage 1: Understand the textbook.

Reads prompts/1_teacher.json, converts the specified PDF pages
from dataset/ to images, and sends them to Kimi K2.6
with JSON Mode enabled. Output is saved to tmp/textbook_understanding_<provider>.json.

Usage (run from code/ with venv activated):
    source .venv/bin/activate
    python scripts/1_understand_textbook.py
    python scripts/1_understand_textbook.py --provider kimi
"""

import argparse
import base64
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# Chinese narration pace (字/min). 14 min ≈ 3780 字, matching the old 3500–4000 band.
RATE_PER_MIN = 270


def duration_to_word_limits(target_minutes):
    """Convert a target video length (minutes) into (lower, upper) character limits.

    Deterministic by design: LLMs don't do arithmetic, so the limits are computed
    here and injected into downstream prompts (stages 3 and 4).
    """
    target = round(target_minutes * RATE_PER_MIN)
    return round(target * 0.90), round(target * 1.10)


def load_prompt(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def pdf_to_base64_images(pdf_path, page_start, page_end, dpi=100):
    """Convert PDF pages (1-indexed, inclusive) to base64-encoded PNG strings."""
    import fitz

    doc = fitz.open(pdf_path)
    total = len(doc)
    print(f"PDF: {total} pages, converting {page_start}-{page_end} at {dpi} DPI", file=sys.stderr)

    images = []
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)

    for i in range(page_start, page_end + 1):
        if i > total:
            print(f"Page {i} exceeds total ({total}), stopping", file=sys.stderr)
            break
        pix = doc[i - 1].get_pixmap(matrix=mat)
        png = pix.tobytes("png")
        b64 = base64.b64encode(png).decode("utf-8")
        images.append(f"data:image/png;base64,{b64}")
        print(f"  Page {i}: {len(png) // 1024}KB", file=sys.stderr)

    doc.close()
    return images


def build_messages(prompt, image_urls):
    """Build the messages list for the multimodal LLM."""
    tb = prompt["textbook"]
    guidelines = "\n".join(f"- {g}" for g in prompt["identity_guidelines"])

    system = (
        f"{prompt['identity']}\n\n"
        f"准则：\n{guidelines}\n\n"
        f"教师教学目标：\n{prompt['teaching_objective']}\n\n"
        f"教材：《{tb['title']}》第 {tb['page_start']}-{tb['page_end']} 页。"
    )

    content = [
        {"type": "text", "text": f"以下是《{tb['title']}》第 {tb['page_start']}-{tb['page_end']} 页的教材内容："}
    ]
    for url in image_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})
    content.append({"type": "text", "text": prompt["rubric"]})

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]


PROVIDERS = {
    "kimi": {
        "model": "kimi-k2.6",
        "api_key_env": "KIMI_API_KEY",
        "api_base_env": "KIMI_API_BASE",
        "api_base_default": "https://api.moonshot.cn/v1",
        "output_name": "textbook_understanding.json",
        # Kimi: don't set temperature/top_p, they're fixed.
        # thinking enabled by default, no extra_body needed.
        "extra_body": None,
        "max_tokens": 100000,
    },
}


def stream_response(client, messages, provider_cfg):
    """Stream the response, printing thinking and content as they arrive.
    Returns the raw content string (should be valid JSON due to JSON Mode)."""
    kwargs = dict(
        model=provider_cfg["model"],
        messages=messages,
        max_tokens=provider_cfg["max_tokens"],
        stream=True,
        response_format={"type": "json_object"},
    )
    if provider_cfg["extra_body"]:
        kwargs["extra_body"] = provider_cfg["extra_body"]

    stream = client.chat.completions.create(**kwargs)

    thinking = False
    content_parts = []

    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta

        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            if not thinking:
                thinking = True
                print("\n--- thinking ---", file=sys.stderr)
            print(reasoning, end="", flush=True, file=sys.stderr)

        if delta.content:
            if thinking:
                thinking = False
                print("\n--- answer ---", file=sys.stderr)
            print(delta.content, end="", flush=True)
            content_parts.append(delta.content)

    print(file=sys.stderr)
    return "".join(content_parts)


def main():
    parser = argparse.ArgumentParser(description="Understand textbook chapter using multimodal LLM")
    parser.add_argument("--provider", choices=["kimi"], default="kimi",
                        help="Which LLM provider to use (default: kimi)")
    args = parser.parse_args()

    cfg = PROVIDERS[args.provider]

    # scripts/ lives under the project root; .env, prompts/, dataset/, tmp/ are siblings
    project_dir = Path(__file__).parent.parent
    load_dotenv(project_dir / ".env")

    api_key = os.environ.get(cfg["api_key_env"], "")
    api_base = os.environ.get(cfg["api_base_env"], cfg["api_base_default"])
    if not api_key:
        print(f"Error: {cfg['api_key_env']} not set in .env", file=sys.stderr)
        sys.exit(1)

    prompt = load_prompt(project_dir / "prompts" / "1_teacher.json")
    tb = prompt["textbook"]

    word_min, word_max = duration_to_word_limits(prompt.get("target_minutes", 14))
    print(f"Target: {prompt.get('target_minutes', 14)} min -> {word_min}-{word_max} zi",
          file=sys.stderr)

    pdf_path = project_dir / "dataset" / tb["pdf_path"]
    if not pdf_path.exists():
        print(f"Error: PDF not found at {pdf_path}", file=sys.stderr)
        sys.exit(1)

    image_urls = pdf_to_base64_images(str(pdf_path), tb["page_start"], tb["page_end"])
    print(f"\n{len(image_urls)} pages ready, calling {cfg['model']}...", file=sys.stderr)

    client = OpenAI(api_key=api_key, base_url=api_base, timeout=600)
    messages = build_messages(prompt, image_urls)

    result = stream_response(client, messages, cfg)

    # Validate JSON before saving
    try:
        parsed = json.loads(result)
    except json.JSONDecodeError as e:
        print(f"\nWarning: JSON parse failed: {e}", file=sys.stderr)
        print("Saving raw output for inspection.", file=sys.stderr)
        parsed = None

    output_path = project_dir / "tmp" / cfg["output_name"]
    output_path.parent.mkdir(exist_ok=True)

    if parsed is not None:
        # Attach the word limits so downstream stages 3 & 4 can read them.
        parsed["_meta"] = {"word_count_min": word_min, "word_count_max": word_max}
        out_text = json.dumps(parsed, ensure_ascii=False, indent=2)
    else:
        out_text = result

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(out_text)

    print(f"\nSaved to {output_path} ({len(out_text)} chars)", file=sys.stderr)


if __name__ == "__main__":
    main()
