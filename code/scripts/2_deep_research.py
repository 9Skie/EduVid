#!/usr/bin/env python3
"""
Stage 2: Deep Research.

Two modes:
  Pipeline mode: Reads textbook understanding from stage 1, uses LLM to
    generate research questions (count set by the prompt), fires deep
    search for each, then summarizes each result.
  Single query mode: Takes a question directly, wraps with source
    constraints, calls deep search.

Usage (run from code/ with venv activated):
    source .venv/bin/activate
    python scripts/2_deep_research.py                                 # full pipeline (alibaba default)
    python scripts/2_deep_research.py "your question here"            # single query
    python scripts/2_deep_research.py "question" --name part1         # single query with custom output name
    python scripts/2_deep_research.py --provider gemini               # full pipeline with Gemini
    python scripts/2_deep_research.py "question" --provider gemini    # single query with Gemini
    python scripts/2_deep_research.py "question" --provider alibaba   # explicit alibaba (same as default)
"""

import os
import re
import sys
import json
import time
import requests
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI

# Load .env
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Deep Search API config
API_URL = "https://dashscope.aliyuncs.com/api/v2/apps/deep-search-agent/chat/completions"
AGENT_ID = os.environ.get("DEEP_SEARCH_AGENT_ID", "aid-7666815243c24c4b896e5a6d6a2ac6df")
AGENT_VERSION = os.environ.get("DEEP_SEARCH_AGENT_VERSION", "beta")

# LLM config for question generation
LLM_MODEL = "deepseek-v4-pro"
LLM_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
LLM_API_BASE = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")

# Load prompt config
_prompt_path = Path(__file__).parent.parent / "prompts" / "2_deep_research.json"
with open(_prompt_path, "r", encoding="utf-8") as _f:
    PROMPT = json.load(_f)

_identity = PROMPT.get("identity", "")
_guidelines = "\n".join(f"- {g}" for g in PROMPT.get("identity_guidelines", []))
_source_constraints = PROMPT.get("source_constraints", "")
RESEARCH_CONTEXT = f"{_identity}\n\n准则：\n{_guidelines}\n\n{_source_constraints}"


def clean_qwen_tags(text):
    """Remove qwen tags but keep the text content. Extract citation URLs."""
    urls = re.findall(r'<qwen:cite\s+url="([^"]+)">', text)

    text = re.sub(r'</?qwen:takeaway[^>]*>', '', text)
    text = re.sub(r'<qwen:cite\s+url="[^"]*">', '', text)
    text = re.sub(r'</qwen:cite>', '', text)
    text = re.sub(r'</?qwen:[^>]*>', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()

    return text, urls


def summarize_research(client, question, raw_content):
    """Send raw deep search output to DeepSeek for a clean summary."""
    summarization_cfg = PROMPT.get("summarization", {})
    system = summarization_cfg.get("identity", "")
    task = summarization_cfg.get("task", "").replace("{question}", question).replace("{raw_content}", raw_content)
    output_format = summarization_cfg.get("output_format", "")

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{task}\n\n{output_format}"},
    ]

    stream = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        max_tokens=100000,
        stream=True,
        reasoning_effort="max",
        extra_body={"thinking": {"type": "enabled"}},
    )

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
                print("  → Summarizing (thinking)...", file=sys.stderr)
            print(reasoning, end="", flush=True, file=sys.stderr)

        if delta.content:
            if thinking:
                thinking = False
                print("\n  → Summarizing (output)...", file=sys.stderr)
            print(delta.content, end="", flush=True, file=sys.stderr)
            content_parts.append(delta.content)

    print(file=sys.stderr)
    return "".join(content_parts)


def call_alibaba_deep_research(question):
    """
    Call the Alibaba Deep Search API with a single question.
    Wraps the question with source constraints.
    Returns (raw_content, reasoning).
    """
    headers = {
        "Authorization": f'Bearer {os.environ.get("DASHSCOPE_API_KEY", "")}',
        "Content-Type": "application/json",
    }

    full_query = f"{RESEARCH_CONTEXT}\n\n研究问题：\n{question}"

    params = {
        "input": {
            "messages": [{"role": "user", "content": full_query}]
        },
        "parameters": {
            "agent_options": {
                "agent_id": AGENT_ID,
                "agent_version": AGENT_VERSION,
            }
        },
        "stream": True,
    }

    response = requests.post(API_URL, headers=headers, json=params, stream=True, timeout=600)

    full_content = ""
    full_reasoning = ""

    for chunk in response.iter_lines():
        if not chunk:
            continue

        chunk_str = chunk.decode("utf-8").strip()
        if not chunk_str.startswith("data:"):
            continue

        json_str = chunk_str[len("data:"):].strip()
        try:
            obj = json.loads(json_str)
        except json.JSONDecodeError:
            continue

        if obj.get("code") and obj.get("code") != "200":
            print(f"  Service error: {obj}", file=sys.stderr)
            continue

        msg = obj.get("output", {}).get("choices", [{}])[0].get("message", {})

        content = msg.get("content", "")
        if isinstance(content, list):
            content = "".join(str(c) for c in content)
        if not content:
            content = msg.get("reasoning_content", "")
            if content:
                if isinstance(content, list):
                    content = "".join(str(c) for c in content)
                full_reasoning += content
        else:
            full_content += content

    return full_content, full_reasoning


def call_gemini_deep_research(question):
    """
    Call the Gemini Deep Research Agent (Interactions API).
    Returns (report_text, citations_list).

    The agent runs asynchronously (background=True) and is polled until
    complete. Output is clean markdown with inline citations.
    """
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in .env")

    client = genai.Client(api_key=api_key)

    full_prompt = f"{RESEARCH_CONTEXT}\n\n研究问题：\n{question}"

    interaction = client.interactions.create(
        input=full_prompt,
        agent="deep-research-preview-04-2026",
        background=True,
    )

    print(f"    Gemini interaction started: {interaction.id}", file=sys.stderr)

    while True:
        interaction = client.interactions.get(interaction.id)
        if interaction.status == "completed":
            break
        elif interaction.status == "failed":
            raise RuntimeError(f"Gemini research failed: {interaction.error}")
        time.sleep(10)

    report_text = ""
    for step in interaction.steps:
        if getattr(step, "type", None) != "model_output":
            continue
        for content_item in step.content:
            if hasattr(content_item, "text") and content_item.text:
                report_text += content_item.text

    urls = re.findall(r'https?://[^\s\]"\)>]+', report_text)
    citations = list(dict.fromkeys(urls))

    return report_text, citations


def call_deep_research(question, provider="alibaba"):
    """Dispatch to the configured deep research provider."""
    if provider == "gemini":
        text, urls = call_gemini_deep_research(question)
        return text, urls, True
    else:
        content, _ = call_alibaba_deep_research(question)
        return content, None, False


def stream_llm_json(client, system, user):
    """Stream an LLM call with JSON Mode. Returns parsed JSON dict."""
    stream = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=100000,
        stream=True,
        response_format={"type": "json_object"},
        reasoning_effort="max",
        extra_body={"thinking": {"type": "enabled"}},
    )

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
            print(delta.content, end="", flush=True, file=sys.stderr)
            content_parts.append(delta.content)

    print(file=sys.stderr)
    return json.loads("".join(content_parts))


def generate_questions(client, textbook_text):
    """Use Kimi to generate research questions from textbook understanding."""
    qg = PROMPT["question_generation"]

    system = f"{_identity}\n\n准则：\n{_guidelines}"
    task = qg["task"].replace("{textbook_understanding}", textbook_text)
    task += "\n\n" + qg["output_format"]
    task += "\n\n" + "\n".join(f"- {c}" for c in qg["constraints"])

    print("Generating research questions from textbook understanding...", file=sys.stderr)
    result = stream_llm_json(client, system, task)
    questions = result.get("questions", [])

    print(f"\nGenerated {len(questions)} questions:", file=sys.stderr)
    for i, q in enumerate(questions, 1):
        print(f"  {i}. {q[:80]}", file=sys.stderr)

    return questions


def save_research_result(question, answer, urls, out_dir, name, raw_answer=None):
    """Save a single research result as JSON."""
    result = {
        "query": question,
        "answer": answer,
        "citations": urls,
    }
    if raw_answer:
        result["raw_answer"] = raw_answer

    out_file = out_dir / f"research_{name}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"  → Saved: {out_file.name} ({len(urls)} citations, {len(answer)} chars)", file=sys.stderr)


def run_pipeline(provider="alibaba"):
    """Full pipeline: read stage 1 output → generate questions → deep research each."""
    project_dir = Path(__file__).parent.parent

    if not LLM_API_KEY:
        print("Error: DEEPSEEK_API_KEY not set in .env", file=sys.stderr)
        sys.exit(1)

    # 1. Read textbook understanding
    textbook_path = project_dir / "tmp" / "textbook_understanding.json"
    if not textbook_path.exists():
        print(f"Error: {textbook_path} not found. Run stage 1 first.", file=sys.stderr)
        sys.exit(1)

    textbook_text = textbook_path.read_text(encoding="utf-8")
    print(f"Loaded textbook understanding: {len(textbook_text)} chars", file=sys.stderr)

    # 2. Generate questions via Kimi
    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_API_BASE, timeout=600)
    questions = generate_questions(client, textbook_text)

    if not questions:
        print("Error: no questions generated", file=sys.stderr)
        sys.exit(1)

    # 3. Deep research + summarize all questions in parallel
    out_dir = project_dir / "tmp"
    out_dir.mkdir(exist_ok=True)

    print(f"\nProvider: {provider}", file=sys.stderr)

    def process_question(idx, question):
        """Worker: deep research → clean → summarize → save. Returns (idx, name, ok)."""
        provider_suffix = "google" if provider == "gemini" else "alibaba"
        name = f"question_{idx}_{provider_suffix}"
        prefix = f"[{idx}/{len(questions)}]"
        try:
            print(f"\n{prefix} Deep research ({provider}) running...", file=sys.stderr)
            content, urls, is_gemini = call_deep_research(question, provider)

            if is_gemini:
                raw_text = content
                summary = raw_text
            else:
                raw_text, urls = clean_qwen_tags(content)
                print(f"{prefix} Summarizing with {LLM_MODEL}...", file=sys.stderr)
                summary = summarize_research(client, question, raw_text)

            save_research_result(question, summary, urls, out_dir, name, raw_answer=raw_text)
            return (idx, name, True)
        except Exception as e:
            print(f"{prefix} FAILED: {e}", file=sys.stderr)
            return (idx, name, False)

    with ThreadPoolExecutor(max_workers=len(questions)) as executor:
        futures = {
            executor.submit(process_question, i, q): i
            for i, q in enumerate(questions, 1)
        }
        results = []
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda r: r[0])
    ok = sum(1 for r in results if r[2])
    print(f"\nDone. {ok}/{len(questions)} research files saved to tmp/", file=sys.stderr)
    for idx, name, success in results:
        status = "OK" if success else "FAILED"
        print(f"  question_{idx}: {status}", file=sys.stderr)


def run_single_query(args, provider="alibaba"):
    """Single query mode: take a question from CLI, call deep search."""

    # Extract --name flag
    name = None
    if "--name" in args:
        idx = args.index("--name")
        if idx + 1 < len(args):
            name = args[idx + 1]
            args = args[:idx] + args[idx + 2:]

    if args and args[0] == "--file":
        filepath = args[1]
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if filepath.endswith(".jsonl"):
            try:
                obj = json.loads(content.splitlines()[0])
                question = obj.get("question", content)
            except json.JSONDecodeError:
                question = content
        else:
            question = content
    else:
        question = " ".join(args)

    print(f"Question: {question[:200]}...", file=sys.stderr)
    print(f"  → Deep research ({provider}) running...", file=sys.stderr)

    content, urls, is_gemini = call_deep_research(question, provider)

    if is_gemini:
        raw_text = content
        summary = raw_text
    else:
        raw_text, urls = clean_qwen_tags(content)
        if LLM_API_KEY:
            client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_API_BASE, timeout=600)
            print(f"  → Summarizing with {LLM_MODEL}...", file=sys.stderr)
            summary = summarize_research(client, question, raw_text)
        else:
            print("  → DEEPSEEK_API_KEY not set, skipping summarization", file=sys.stderr)
            summary = raw_text

    out_dir = Path(__file__).parent.parent / "tmp"
    out_dir.mkdir(exist_ok=True)

    if name:
        out_name = name
    else:
        out_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_research_result(question, summary, urls, out_dir, out_name, raw_answer=raw_text)
    print(f"\n--- Done ---", file=sys.stderr)


def main():
    args = sys.argv[1:]

    # Extract --provider flag
    provider = "alibaba"
    if "--provider" in args:
        idx = args.index("--provider")
        if idx + 1 < len(args):
            provider = args[idx + 1]
            args = args[:idx] + args[idx + 2:]

    if provider not in ("alibaba", "gemini"):
        print(f"Error: unknown provider '{provider}'. Use 'alibaba' or 'gemini'.", file=sys.stderr)
        sys.exit(1)

    if not args:
        run_pipeline(provider)
    else:
        run_single_query(args, provider)


if __name__ == "__main__":
    main()
