#!/usr/bin/env python3
"""
Stage 3: Analysis.

DeepSeek reads the textbook understanding + research summaries and outputs a
detailed Markdown teaching blueprint (analysis.md). A deterministic code parser
then splits that Markdown into structured JSON (analysis.json) — no second LLM
call, so no paragraph can be silently dropped. The Markdown is deleted after
parsing; the JSON is what downstream stages consume.

Usage (run from code/ with venv activated):
    source .venv/bin/activate
    python scripts/3_research_analysis.py
    python scripts/3_research_analysis.py --textbook tmp/textbook_understanding.json --research tmp/research_*.json
"""

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


PROVIDER = {
    "model": "deepseek-v4-pro",
    "api_key_env": "DEEPSEEK_API_KEY",
    "api_base_env": "DEEPSEEK_API_BASE",
    "api_base_default": "https://api.deepseek.com/v1",
    "max_tokens": 100000,
    "reasoning_effort": "max",
    "extra_body": {"thinking": {"type": "enabled"}},
}


def load_prompt(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_textbook_with_limits(path):
    """Load the stage 1 textbook digest, extracting word-count limits from _meta.

    Returns (digest_text_without_meta, word_min, word_max). Falls back to the
    14-minute band (3402-4158) if _meta is missing or the file isn't JSON.
    """
    raw = load_text(path)
    default_min, default_max = 3402, 4158
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return raw, default_min, default_max
    meta = obj.pop("_meta", {})
    word_min = meta.get("word_count_min", default_min)
    word_max = meta.get("word_count_max", default_max)
    digest = json.dumps(obj, ensure_ascii=False, indent=2)
    return digest, word_min, word_max


def _fill(text, word_min, word_max):
    """Substitute {word_count_min}/{word_count_max} placeholders."""
    return text.replace("{word_count_min}", str(word_min)).replace("{word_count_max}", str(word_max))


def collect_research(research_files):
    """Load all research JSON files and combine into a single text block."""
    parts = []
    for path in sorted(research_files):
        raw = load_text(path)
        try:
            obj = json.loads(raw)
            parts.append(f"### {Path(path).stem}\n查询: {obj.get('query', '?')}\n\n{obj.get('answer', '')}")
        except json.JSONDecodeError:
            parts.append(f"### {Path(path).stem}\n{raw}")
    return "\n\n---\n\n".join(parts)


def stream_call(client, messages, use_json=False):
    """Stream one LLM call. Returns the full text content."""
    kwargs = dict(
        model=PROVIDER["model"],
        messages=messages,
        max_tokens=PROVIDER["max_tokens"],
        stream=True,
        reasoning_effort=PROVIDER["reasoning_effort"],
        extra_body=PROVIDER["extra_body"],
    )
    if use_json:
        kwargs["response_format"] = {"type": "json_object"}

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
                print("\n--- output ---", file=sys.stderr)
            print(delta.content, end="", flush=True)
            content_parts.append(delta.content)

    print(file=sys.stderr)
    return "".join(content_parts)


def round_analysis(client, system, user):
    """Round 1: Generate markdown analysis from textbook + research."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return stream_call(client, messages, use_json=False)


def parse_analysis_markdown(md):
    """Parse the fixed-header Markdown analysis into a structured dict.

    Lossless: splits on the '## ' headers the prompt requires, preserving every
    word of prose in the content fields. Replaces the old LLM markdown->JSON round.
    """
    result = {"summary": "", "narrative_line": "", "blocks": []}

    # Split on level-2 headers ('## ' but not '### ').
    for section in re.split(r'^## ', md, flags=re.MULTILINE)[1:]:
        header, _, body = section.partition("\n")
        header = header.strip()
        body = body.strip()

        if header == "总览":
            if "### 叙事线" in body:
                summary, _, narrative = body.partition("### 叙事线")
                result["summary"] = summary.strip()
                result["narrative_line"] = narrative.strip()
            else:
                result["summary"] = body
        elif header.startswith("知识板块："):
            name = header[len("知识板块："):].strip()
            content, tb_pts, res_pts = _peel_source_lines(body)
            result["blocks"].append({
                "name": name,
                "content": content,
                "textbook_points": tb_pts,
                "research_points": res_pts,
            })

    return result


def _peel_source_lines(body):
    """Extract trailing 教材依据：/研究补充： lines; return (content, textbook_pts, research_pts)."""
    textbook_pts = ""
    research_pts = ""

    m = re.search(r'^教材依据：(.*)$', body, re.MULTILINE)
    if m:
        textbook_pts = m.group(1).strip()
    m = re.search(r'^研究补充：(.*)$', body, re.MULTILINE)
    if m:
        research_pts = m.group(1).strip()

    content = re.sub(r'^教材依据：.*$\n?', '', body, flags=re.MULTILINE)
    content = re.sub(r'^研究补充：.*$\n?', '', content, flags=re.MULTILINE)
    return content.strip(), textbook_pts, research_pts


def main():
    parser = argparse.ArgumentParser(description="Compose textbook + research into analysis (markdown → JSON)")
    parser.add_argument("--textbook", default=None,
                        help="Path to textbook understanding JSON (default: tmp/textbook_understanding.json)")
    parser.add_argument("--research", nargs="+", default=None,
                        help="Paths to research JSON files (default: tmp/research_*.json)")
    args = parser.parse_args()

    project_dir = Path(__file__).parent.parent
    load_dotenv(project_dir / ".env")

    api_key = os.environ.get(PROVIDER["api_key_env"], "")
    api_base = os.environ.get(PROVIDER["api_base_env"], PROVIDER["api_base_default"])
    if not api_key:
        print(f"Error: {PROVIDER['api_key_env']} not set in .env", file=sys.stderr)
        sys.exit(1)

    # Resolve input paths
    textbook_path = args.textbook or str(project_dir / "tmp" / "textbook_understanding.json")
    research_paths = args.research or sorted(glob.glob(str(project_dir / "tmp" / "research_*.json")))

    if not os.path.exists(textbook_path):
        print(f"Error: Textbook understanding not found at {textbook_path}", file=sys.stderr)
        print("Run scripts/1_understand_textbook.py first.", file=sys.stderr)
        sys.exit(1)

    if not research_paths:
        print(f"Error: No research files found in tmp/", file=sys.stderr)
        print("Run scripts/2_deep_research.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Textbook: {textbook_path}", file=sys.stderr)
    print(f"Research files: {len(research_paths)}", file=sys.stderr)

    textbook_text, word_min, word_max = load_textbook_with_limits(textbook_path)
    research_text = collect_research(research_paths)

    prompt = load_prompt(project_dir / "prompts" / "3_research_analysis.json")
    client = OpenAI(api_key=api_key, base_url=api_base, timeout=600)

    tmp_dir = project_dir / "tmp"

    # --- Round 1: Markdown analysis ---
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"ROUND 1: ANALYSIS (markdown)", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    system = _fill(prompt["identity"], word_min, word_max)
    guidelines = "\n".join(f"- {g}" for g in prompt["identity_guidelines"])
    framework = prompt.get("narrative_framework", {})
    framework_text = "\n".join(f"  {k}：{v}" for k, v in framework.items())
    system = f"{system}\n\n准则：\n{guidelines}\n\n叙事结构（dream / expansion / mellow / overlong）：\n{framework_text}"

    task = _fill(prompt["task"], word_min, word_max)
    task = task.replace("{textbook_understanding}", textbook_text)
    task = task.replace("{deep_research_results}", research_text)
    task += "\n\n" + prompt["output_format"]
    task += "\n\n" + "\n".join(_fill(c, word_min, word_max) for c in prompt["constraints"])

    print(f"Word band: {word_min}-{word_max} zi", file=sys.stderr)

    md_result = round_analysis(client, system, task)

    md_path = tmp_dir / "analysis.md"
    md_path.write_text(md_result, encoding="utf-8")
    print(f"\nMarkdown saved: {md_path} ({len(md_result)} chars)", file=sys.stderr)

    # --- Parse Markdown -> structured JSON (deterministic, no LLM) ---
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"PARSING MARKDOWN -> JSON", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    analysis = parse_analysis_markdown(md_result)
    json_path = tmp_dir / "analysis.json"
    json_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"JSON saved: {json_path}", file=sys.stderr)
    print(f"  summary: {len(analysis['summary'])} chars", file=sys.stderr)
    print(f"  narrative_line: {len(analysis['narrative_line'])} chars", file=sys.stderr)
    print(f"  blocks: {len(analysis['blocks'])}", file=sys.stderr)

    md_path.unlink()
    print(f"Deleted {md_path.name} (Markdown consumed by parser)", file=sys.stderr)


if __name__ == "__main__":
    main()
