#!/usr/bin/env python3
"""
Stage 7: Script chapter → clip plan.

Per-chapter. Reads one chapter's timed lines (from stage 6) plus the cast
entries relevant to that chapter (from stage 5), sends them to Kimi K2.6 with
the clip-plan prompt, and saves a clip plan JSON with computed per-clip timings.

Input:
  - tmp/audio/voice_chapter_{N}_{voice}_lines.json   (sentence-level timings)
  - casts/{fixed,consistent,chapters/chapter_N}/      (cast entries)
  - tmp/script_final.json                             (chapter names)

Output:
  - tmp/clip_plans/clip_plan_chapter_{N}.json

Usage (run from code/ with venv activated):
    python scripts/7_script_chapter_to_clip_plan.py --chapter 1   # voice = narration_gender
    python scripts/7_script_chapter_to_clip_plan.py               # all chapters
    python scripts/7_script_chapter_to_clip_plan.py --voice male  # override the voice
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

PROVIDERS = {
    "kimi": {
        "model": "kimi-k2.6",
        "api_key_env": "KIMI_API_KEY",
        "api_base_env": "KIMI_API_BASE",
        "api_base_default": "https://api.moonshot.cn/v1",
        "max_tokens": 100000,
        "extra_body": None,  # thinking enabled by default on Kimi
    },
}


# ── I/O helpers ───────────────────────────────────────────────────────

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(data, path):
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ── Input assembly ────────────────────────────────────────────────────

def load_timed_lines(audio_dir, chapter, voice):
    """Load sentence-level lines, assign line_ids, convert seconds → ms."""
    lines_path = audio_dir / f"voice_chapter_{chapter}_{voice}_lines.json"
    if not lines_path.exists():
        raise FileNotFoundError(
            f"Timed lines not found: {lines_path}\nRun stage 6 first."
        )
    raw = load_json(lines_path)
    lines = []
    for i, entry in enumerate(raw, 1):
        lines.append({
            "line_id": f"line_{i}",
            "text": entry["text"],
            "start_ms": round(entry["start"] * 1000),
            "end_ms": round(entry["end"] * 1000),
        })
    return lines


def _load_cast_dir(dir_path):
    """Load every *.json cast entry from a flat directory."""
    entries = []
    if not dir_path.exists():
        return entries
    for json_file in sorted(dir_path.glob("*.json")):
        try:
            entries.append(load_json(json_file))
        except (json.JSONDecodeError, KeyError):
            print(f"  skip malformed cast file: {json_file.name}", file=sys.stderr)
    return entries


def gather_cast(casts_dir, chapter):
    """Recurring cast (fixed anchors + consistent) + chapter-specific cast.

    Teacher/students are presented as ordinary recurring cast, not a special
    group — they're actors like everyone else.
    """
    return {
        "贯穿全片的角色": (
            _load_cast_dir(casts_dir / "fixed")
            + _load_cast_dir(casts_dir / "consistent")
        ),
        "本章出现的角色": _load_cast_dir(
            casts_dir / "chapters" / f"chapter_{chapter}"
        ),
    }


def format_timed_script(lines):
    """Render timed lines so the model sees line_id + timing + text."""
    return "\n".join(
        f"{ln['line_id']}  [{ln['start_ms']}ms – {ln['end_ms']}ms]  {ln['text']}"
        for ln in lines
    )


def format_cast_set(groups):
    """Render cast entries grouped by availability scope."""
    blocks = []
    for group_name, entries in groups.items():
        if not entries:
            continue
        rows = [f"【{group_name}】"]
        for e in entries:
            cid = e.get("cast_id", "?")
            name = e.get("name", "?")
            kind = e.get("kind", "?")
            desc = e.get("description", "")
            rows.append(f"[{cid}] {name}（{kind}）: {desc}")
        blocks.append("\n".join(rows))
    return "\n\n".join(blocks) if blocks else "（本章无额外 cast）"


# ── Prompt + LLM ──────────────────────────────────────────────────────

def build_system_prompt(prompt):
    """Assemble rules and format into the system message."""
    constraints = "\n".join(f"- {c}" for c in prompt["constraints"])
    parts = [
        prompt["identity"],
        "## 切分规则\n" + prompt["segmentation_rules"],
        "## 工具选择（routing）\n" + prompt["routing"],
        "## 描述要求\n" + prompt["description_guidance"],
    ]
    if prompt.get("voiceover_rule"):
        parts.append("## 旁白与人物（全片通则）\n" + prompt["voiceover_rule"])
    parts.append("## 输出格式\n" + prompt["output_format"])
    parts.append("## 约束\n" + constraints)
    return "\n\n".join(parts)


def build_user_message(prompt, timed_script_text, cast_set_text):
    """Fill the task template with real data."""
    return (
        prompt["task"]
        .replace("{timed_script}", timed_script_text)
        .replace("{cast_set}", cast_set_text)
    )


def stream_response(client, messages, provider_cfg):
    """Stream Kimi response. Thinking → stderr, content → stdout. Returns content string."""
    kwargs = dict(
        model=provider_cfg["model"],
        messages=messages,
        max_tokens=provider_cfg["max_tokens"],
        stream=True,
        response_format={"type": "json_object"},
    )
    if provider_cfg.get("extra_body"):
        kwargs["extra_body"] = provider_cfg["extra_body"]

    stream = client.chat.completions.create(**kwargs)
    content_parts = []

    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            print(reasoning, end="", flush=True, file=sys.stderr)
        if delta.content:
            content_parts.append(delta.content)
            print(delta.content, end="", flush=True)

    print(file=sys.stderr)  # newline after thinking stream
    return "".join(content_parts)


# ── Post-processing ───────────────────────────────────────────────────

def enrich_clip_timings(clips, lines_by_id):
    """Derive start_ms / end_ms / duration_ms for each clip from its lines."""
    for clip in clips:
        starts, ends = [], []
        for lid in clip.get("line_ids", []):
            ln = lines_by_id.get(lid)
            if ln:
                starts.append(ln["start_ms"])
                ends.append(ln["end_ms"])
        if starts and ends:
            clip["start_ms"] = min(starts)
            clip["end_ms"] = max(ends)
            clip["duration_ms"] = clip["end_ms"] - clip["start_ms"]
    return clips


def validate_clip_plan(clips, expected_line_ids, available_cast_ids):
    """Check coverage, ordering, cast references, and method validity."""
    warnings = []

    # Flatten line_ids across clips in declared order
    flat = []
    for clip in clips:
        flat.extend(clip.get("line_ids", []))

    if flat != expected_line_ids:
        seen, expected = set(flat), set(expected_line_ids)
        missing = expected - seen
        extra = seen - expected
        if missing:
            warnings.append(
                f"覆盖不全：{len(missing)} 行未被任何 clip 覆盖: {sorted(missing)[:5]}"
            )
        if extra:
            warnings.append(
                f"多余 line_id：{len(extra)} 个不在脚本中: {sorted(extra)[:5]}"
            )
        if not missing and not extra:
            warnings.append("顺序不一致：clip 内 line_ids 与脚本顺序不符")
        # detect a line reused across clips
        dupes = [lid for lid, c in Counter(flat).items() if c > 1]
        if dupes:
            warnings.append(f"行重复：{dupes[:5]} 出现在多个 clip 中")

    for clip in clips:
        for cid in clip.get("cast_ids", []):
            if cid not in available_cast_ids:
                warnings.append(
                    f"clip {clip.get('clip_id', '?')}: cast_id '{cid}' 不在 cast 列表中"
                )

    valid_methods = {"seedance", "manim", "remotion"}
    for clip in clips:
        m = clip.get("method")
        if m not in valid_methods:
            warnings.append(
                f"clip {clip.get('clip_id', '?')}: 无效 method '{m}'"
            )

    return warnings


# ── Per-chapter processing ────────────────────────────────────────────

def process_chapter(client, provider_cfg, prompt, chapter_num, chapter_name,
                    audio_dir, casts_dir, voice, out_dir):
    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"Chapter {chapter_num}: {chapter_name}", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)

    timed_lines = load_timed_lines(audio_dir, chapter_num, voice)
    print(f"Loaded {len(timed_lines)} timed lines", file=sys.stderr)

    cast_groups = gather_cast(casts_dir, chapter_num)
    total_cast = sum(len(v) for v in cast_groups.values())
    print(f"Gathered {total_cast} cast entries", file=sys.stderr)

    system = build_system_prompt(prompt)
    user = build_user_message(
        prompt,
        format_timed_script(timed_lines),
        format_cast_set(cast_groups),
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    print(f"Calling Kimi ({provider_cfg['model']})...", file=sys.stderr)
    t0 = time.time()
    raw = stream_response(client, messages, provider_cfg)
    elapsed = time.time() - t0

    try:
        plan = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"Error: clip plan JSON parse failed: {e}", file=sys.stderr)
        debug_path = out_dir / f"clip_plan_chapter_{chapter_num}_raw.txt"
        debug_path.write_text(raw, encoding="utf-8")
        print(f"Raw output saved to {debug_path}", file=sys.stderr)
        return False

    clips = plan.get("clips", [])
    lines_by_id = {ln["line_id"]: ln for ln in timed_lines}
    clips = enrich_clip_timings(clips, lines_by_id)

    expected = [ln["line_id"] for ln in timed_lines]
    available_cast = {
        e.get("cast_id")
        for entries in cast_groups.values()
        for e in entries
    }
    warnings = validate_clip_plan(clips, expected, available_cast)
    if warnings:
        print(f"\n⚠ 校验警告:", file=sys.stderr)
        for w in warnings:
            print(f"  - {w}", file=sys.stderr)

    output = {
        "chapter": chapter_num,
        "chapter_name": chapter_name,
        "voice": voice,
        "model": provider_cfg["model"],
        "line_count": len(timed_lines),
        "clip_count": len(clips),
        "elapsed_s": round(elapsed, 1),
        "clips": clips,
    }
    out_path = out_dir / f"clip_plan_chapter_{chapter_num}.json"
    save_json(output, out_path)

    method_counts = Counter(c.get("method", "?") for c in clips)
    method_str = ", ".join(f"{m}={n}" for m, n in sorted(method_counts.items()))
    print(
        f"\nSaved: {out_path} ({len(clips)} clips [{method_str}], {elapsed:.1f}s)",
        file=sys.stderr,
    )
    return True


# ── Main ──────────────────────────────────────────────────────────────

def resolve_voice(project_dir, cli_voice):
    """Voice follows the teacher's chosen narration gender unless overridden."""
    if cli_voice:
        return cli_voice
    try:
        cfg = json.loads(
            (project_dir / "prompts" / "1_teacher.json").read_text(encoding="utf-8")
        )
        g = cfg.get("narration_gender", "female")
        return g if g in ("male", "female") else "female"
    except Exception:
        return "female"


def main():
    parser = argparse.ArgumentParser(
        description="Stage 7: Script chapter → clip plan"
    )
    parser.add_argument(
        "--chapter", type=int, default=None,
        help="Chapter number (1-indexed). If omitted, processes all chapters.",
    )
    parser.add_argument(
        "--voice", choices=["female", "male"], default=None,
        help="Override voice (default: teacher's narration_gender).",
    )
    parser.add_argument(
        "--prompt", default="prompts/7_script_chapter_to_clip_plan_oneshot.json",
        help="Path to clip plan prompt JSON.",
    )
    parser.add_argument(
        "--script", default="tmp/script_final.json",
        help="Path to final script JSON (for chapter names).",
    )
    parser.add_argument(
        "--out-dir", default="tmp/clip_plans",
        help="Output directory for clip plan JSONs.",
    )
    args = parser.parse_args()

    project_dir = Path(__file__).parent.parent
    load_dotenv(project_dir / ".env")
    args.voice = resolve_voice(project_dir, args.voice)

    provider = PROVIDERS["kimi"]
    api_key = os.environ.get(provider["api_key_env"], "")
    if not api_key:
        print(f"Error: {provider['api_key_env']} not set in .env", file=sys.stderr)
        sys.exit(1)
    api_base = os.environ.get(
        provider["api_base_env"], provider["api_base_default"]
    )
    client = OpenAI(api_key=api_key, base_url=api_base)

    prompt_path = project_dir / args.prompt
    if not prompt_path.exists():
        print(f"Error: prompt not found: {prompt_path}", file=sys.stderr)
        sys.exit(1)
    prompt = load_json(prompt_path)

    audio_dir = project_dir / "tmp" / "audio"
    casts_dir = project_dir / "casts"
    out_dir = project_dir / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve chapter list
    script_path = project_dir / args.script
    if script_path.exists():
        script_data = load_json(script_path)
        chapters = [
            (i + 1, ch["name"])
            for i, ch in enumerate(script_data.get("chapters", []))
        ]
    else:
        if args.chapter is None:
            print(f"Error: {script_path} not found and no --chapter given", file=sys.stderr)
            sys.exit(1)
        chapters = [(args.chapter, f"Chapter {args.chapter}")]

    if args.chapter is not None:
        chapters = [(n, name) for n, name in chapters if n == args.chapter]
        if not chapters:
            print(f"Error: chapter {args.chapter} not found in {script_path}", file=sys.stderr)
            sys.exit(1)

    failed = []
    for chapter_num, chapter_name in chapters:
        ok = process_chapter(
            client, provider, prompt, chapter_num, chapter_name,
            audio_dir, casts_dir, args.voice, out_dir,
        )
        if not ok:
            failed.append(chapter_num)

    if failed:
        print(f"\nFailed chapters: {failed}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
