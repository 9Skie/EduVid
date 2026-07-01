#!/usr/bin/env python3
"""
Stage 8: Clip plan → visual flow

One-shot LLM call per chapter. Takes all clips (segmented + routed + described
from stage 7) plus per-character audio timings, cast descriptions, and art
style. Produces per-clip visual designs:
  - seedance       → single shot prompt + cast_ids
  - manim/remotion → time-anchored beats synced to narration

Input:
  - tmp/clip_plans/clip_plan_chapter_N.json      (from stage 7)
  - tmp/audio/voice_chapter_N_*_alignment.json    (from stage 6)
  - tmp/audio/voice_chapter_N_*_lines.json        (from stage 6)
  - casts/{fixed,consistent,chapters/chapter_N}/  (from stage 5)
  - prompts/art_style.json

Output:
  - tmp/visual_flow/visual_flow_chapter_N.json

Usage (run from code/ with venv activated):
    python scripts/8_clip_plan_to_visual_flow.py --chapter 1
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

KIMI_CFG = {
    "model": "kimi-k2.6",
    "api_key_env": "KIMI_API_KEY",
    "api_base_env": "KIMI_API_BASE",
    "api_base_default": "https://api.moonshot.cn/v1",
    "max_tokens": 100000,
    "extra_body": None,
}


# ── I/O helpers ───────────────────────────────────────────────────────

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(data, path):
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ── Character grouping ────────────────────────────────────────────────

def group_chars_by_line(lines, alignment):
    """Assign each alignment character to its line (1-indexed → line_N).

    Walks through alignment characters sequentially, consuming
    len(line.text) characters per line. Verified: concatenated line texts
    match the alignment characters array exactly.
    """
    chars = alignment["characters"]
    starts = alignment["char_start_times"]
    ends = alignment["char_end_times"]

    by_line = {}
    idx = 0
    for i, line in enumerate(lines, 1):
        lid = f"line_{i}"
        n = len(line["text"])
        by_line[lid] = [
            {"char": chars[j], "start": starts[j], "end": ends[j]}
            for j in range(idx, min(idx + n, len(chars)))
        ]
        idx += n
    return by_line


def gather_clip_chars(clips, chars_by_line):
    """Collect per-character timings for each clip from its line_ids."""
    for clip in clips:
        clip_chars = []
        for lid in clip.get("line_ids", []):
            clip_chars.extend(chars_by_line.get(lid, []))
        clip["_char_timings"] = clip_chars
    return clips


# ── Formatting ────────────────────────────────────────────────────────

def format_clips_block(clips):
    """Build a text block showing each clip with its per-character timings.

    Every character is shown with its exact start-end window. No data
    is dropped or summarized.
    """
    blocks = []
    for clip in clips:
        clip_id = clip.get("clip_id", "?")
        method = clip.get("method", "?")
        description = clip.get("description", "")
        cast_ids = clip.get("cast_ids", [])
        line_ids = clip.get("line_ids", [])
        char_timings = clip.get("_char_timings", [])

        header = f"### {clip_id} (method: {method})"
        if cast_ids:
            header += f"  cast: {', '.join(cast_ids)}"

        lines_parts = []
        # Split char timings back into lines for readability
        for lid in line_ids:
            line_chars = [c for c in char_timings if c.get("_line") == lid]

        # Simpler: just format all chars for this clip in sequence
        if char_timings:
            char_str = " ".join(
                f'{c["char"]}({c["start"]:.2f}-{c["end"]:.2f})'
                for c in char_timings
            )
            clip_start = char_timings[0]["start"]
            clip_end = char_timings[-1]["end"]
            lines_parts.append(
                f"  lines {line_ids[0]}–{line_ids[-1]} "
                f"[{clip_start:.2f}-{clip_end:.2f}s]:\n  {char_str}"
            )

        blocks.append(
            f"{header}\n"
            f"  description: {description}\n"
            + "\n".join(lines_parts)
        )
    return "\n\n".join(blocks)


def format_cast_block(cast_groups):
    """Render cast entries grouped by availability scope."""
    blocks = []
    for group_name, entries in cast_groups.items():
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
    return "\n\n".join(blocks) if blocks else "（无 cast）"


# ── Cast loading ──────────────────────────────────────────────────────

def _load_cast_dir(dir_path):
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
    # Teacher/students = ordinary recurring cast, merged with consistent.
    return {
        "贯穿全片的角色": (
            _load_cast_dir(casts_dir / "fixed")
            + _load_cast_dir(casts_dir / "consistent")
        ),
        "本章出现的角色": _load_cast_dir(
            casts_dir / "chapters" / f"chapter_{chapter}"
        ),
    }


# ── Prompt building ───────────────────────────────────────────────────

def build_system_prompt(prompt):
    constraints = "\n".join(f"- {c}" for c in prompt.get("constraints", []))
    parts = [prompt["identity"]]
    if prompt.get("design_by_method"):
        parts.append("## 设计形式（按工具分）\n" + prompt["design_by_method"])
    if prompt.get("beats_guidance"):
        parts.append("## 节拍（beats）规则\n" + prompt["beats_guidance"])
    if prompt.get("shot_guidance"):
        parts.append("## 镜头（shot）规则\n" + prompt["shot_guidance"])
    if prompt.get("voiceover_rule"):
        parts.append("## 旁白与人物（全片通则）\n" + prompt["voiceover_rule"])
    parts.append("## 输出格式\n" + prompt["output_format"])
    if constraints:
        parts.append("## 约束\n" + constraints)
    return "\n\n".join(parts)


def build_user_message(prompt, clips_text, cast_text, art_style_text):
    return (
        prompt["task"]
        .replace("{clips}", clips_text)
        .replace("{cast_set}", cast_text)
        .replace("{art_style}", art_style_text)
    )


# ── LLM call ──────────────────────────────────────────────────────────

def stream_response(client, messages, cfg):
    kwargs = dict(
        model=cfg["model"],
        messages=messages,
        max_tokens=cfg["max_tokens"],
        stream=True,
        response_format={"type": "json_object"},
    )
    if cfg.get("extra_body"):
        kwargs["extra_body"] = cfg["extra_body"]

    stream = client.chat.completions.create(**kwargs)
    content_parts = []
    reasoning_chars = 0
    finish_reason = None

    for chunk in stream:
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        if choice.finish_reason:
            finish_reason = choice.finish_reason
        delta = choice.delta
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            reasoning_chars += len(reasoning)
            print(reasoning, end="", flush=True, file=sys.stderr)
        if delta.content:
            content_parts.append(delta.content)
            print(delta.content, end="", flush=True)

    print(file=sys.stderr)
    content = "".join(content_parts)

    # A thinking model can spend the whole max_tokens budget on reasoning and
    # emit little/no JSON. Surface that explicitly instead of letting it fall
    # through to a cryptic JSONDecodeError.
    if finish_reason == "length":
        print(
            f"\n⚠ 输出被 max_tokens={cfg['max_tokens']} 截断（finish_reason=length）："
            f"思考约 {reasoning_chars} 字，正文 JSON 仅 {len(content)} 字——"
            f"预算几乎都花在思考上，JSON 未完整。",
            file=sys.stderr,
        )
    elif not content.strip():
        print(
            f"\n⚠ 模型只输出了思考（约 {reasoning_chars} 字）、没有正文 JSON"
            f"（finish_reason={finish_reason}）。",
            file=sys.stderr,
        )

    return content


# ── Validation ────────────────────────────────────────────────────────

def validate_designs(designs_by_id, clips):
    warnings = []

    for clip in clips:
        cid = clip.get("clip_id", "?")
        design = designs_by_id.get(cid)

        if not design:
            warnings.append(f"{cid}: 缺少 visual design")
            continue

        kind = design.get("kind")
        method = clip.get("method", "?")

        if method == "seedance":
            if kind != "shot":
                warnings.append(f"{cid}: method=seedance 但 kind={kind}（应为 shot）")
            if not design.get("prompt"):
                warnings.append(f"{cid}: shot 缺少 prompt")

        elif method in ("manim", "remotion"):
            if kind != "beats":
                warnings.append(f"{cid}: method={method} 但 kind={kind}（应为 beats）")
            beats = design.get("beats", [])
            if not beats:
                warnings.append(f"{cid}: beats 为空")

            # Check beat times are within clip's absolute time range
            clip_start_s = clip.get("start_ms", 0) / 1000.0
            clip_end_s = clip.get("end_ms", 0) / 1000.0
            for i, beat in enumerate(beats):
                t = beat.get("t", [])
                if len(t) != 2:
                    warnings.append(f"{cid}: beat {i} 的 t 不是 [start, end]")
                    continue
                beat_start, beat_end = t[0], t[1]
                # beats use absolute timestamps (matching char_timings)
                if beat_start < clip_start_s - 0.5 or beat_end > clip_end_s + 0.5:
                    warnings.append(
                        f"{cid}: beat {i} 时间 [{beat_start}-{beat_end}] "
                        f"超出 clip 绝对范围 [{clip_start_s:.1f}-{clip_end_s:.1f}]"
                    )

            # Gapless coverage: beats must tile [start, end] — no gaps / overlap.
            valid_ts = [
                b["t"] for b in beats
                if isinstance(b.get("t"), list) and len(b["t"]) == 2
            ]
            if valid_ts:
                ordered = sorted(valid_ts, key=lambda t: t[0])
                if abs(ordered[0][0] - clip_start_s) > 0.25:
                    warnings.append(
                        f"{cid}: 首个 beat 未从 clip 起点开始 "
                        f"（{ordered[0][0]:.2f} vs {clip_start_s:.2f}）"
                    )
                if abs(ordered[-1][1] - clip_end_s) > 0.25:
                    warnings.append(
                        f"{cid}: 末个 beat 未到 clip 终点 "
                        f"（{ordered[-1][1]:.2f} vs {clip_end_s:.2f}）"
                    )
                for i in range(len(ordered) - 1):
                    gap = ordered[i + 1][0] - ordered[i][1]
                    if abs(gap) > 0.25:
                        kind_w = "间隙" if gap > 0 else "重叠"
                        warnings.append(
                            f"{cid}: beat 间{kind_w} {gap:+.2f}s "
                            f"（{ordered[i][1]:.2f}→{ordered[i + 1][0]:.2f}）"
                        )

    return warnings


# ── Seedance segment splitting ────────────────────────────────────────
#
# Seedance clips must be 2-15s. Stage 7 isn't told this (so it can think in
# whole scenes), so stage 8 splits long seedance shots post-hoc:
#   - cut COUNT (N) and ideal positions: programmatic (equal division)
#   - cut POSITION (which char boundary): LLM picks, for semantic naturalness
#   - hard [2s, 15s] guard: programmatic; falls back to equal split if the
#     LLM's choice is illegal
# Per-segment motion prompts are NOT derived here — that needs real generated
# frames and happens in stage 9 (segment k's prompt derived from seg k-1's
# last frame). Stage 8 only emits segment boundaries + narration chunks.

SEEDANCE_MIN_S = 2.0
SEEDANCE_MAX_S = 15.0
_SNIPPET_RADIUS = 10


def _clip_bounds_s(clip):
    return clip["start_ms"] / 1000.0, clip["end_ms"] / 1000.0


def _text_of(timings):
    return "".join(c["char"] for c in timings)


def _char_boundary_times(timings):
    return sorted({round(c["end"], 3) for c in timings})


def _build_cut_plan(clip):
    """For one long seedance clip: compute N and per-cut candidate boundaries."""
    timings = clip["char_timings"]
    text = _text_of(timings)
    start_s, end_s = _clip_bounds_s(clip)
    D = end_s - start_s
    N = max(2, math.ceil(D / SEEDANCE_MAX_S))

    idx_bounds = [(i, c["end"]) for i, c in enumerate(timings)
                  if i + 1 < len(timings)]

    cuts = []
    for k in range(1, N):
        ideal = start_s + k * D / N
        W = min(D / N * 0.4, 3.0)
        lo = max(start_s + SEEDANCE_MIN_S, ideal - W)
        hi = min(end_s - SEEDANCE_MIN_S, ideal + W)
        in_win = sorted(((ci, t) for ci, t in idx_bounds if lo <= t <= hi),
                        key=lambda x: x[1])
        if len(in_win) > 6:
            step = len(in_win) / 6.0
            in_win = [in_win[min(len(in_win) - 1, int(i * step))]
                      for i in range(6)]
        candidates = []
        for j, (ci, t) in enumerate(in_win):
            before = text[max(0, ci - _SNIPPET_RADIUS + 1):ci + 1]
            after = text[ci + 1:ci + 1 + _SNIPPET_RADIUS]
            candidates.append({
                "id": f"cand_{chr(ord('a') + j)}",
                "t": round(t, 2),
                "before": before,
                "after": after,
            })
        cuts.append({
            "k": k,
            "ideal": round(ideal, 2),
            "lo": round(lo, 2),
            "hi": round(hi, 2),
            "candidates": candidates,
        })
    return {"clip_id": clip["clip_id"], "N": N, "cuts": cuts}


def _build_split_message(plans):
    system = (
        "你的任务是为视频片段的旁白选择最自然的语义切分点。"
        "你会看到每个片段需要做的若干处切分，每处给出几个候选位置"
        "（每个候选显示切分处前后的文字和时间戳）。"
        "请为每处切分选择语义上最自然的候选——优先句末（。！？）或分句末（，；、），"
        "让每段旁白尽量是一个完整的语意单元。只能从给出的候选中选择，返回候选 id。"
    )
    blocks = []
    for plan in plans:
        lines = [f"片段 {plan['clip_id']}，共需 {len(plan['cuts'])} 处切分："]
        for cut in plan["cuts"]:
            if not cut["candidates"]:
                continue
            lines.append(
                f"  切分 {cut['k']}（理想 {cut['ideal']}s，范围 "
                f"{cut['lo']}-{cut['hi']}s）："
            )
            for cand in cut["candidates"]:
                lines.append(
                    f"    {cand['id']}（{cand['t']}s）："
                    f"「…{cand['before']}」|「{cand['after']}…」"
                )
        blocks.append("\n".join(lines))
    user = (
        "\n\n".join(blocks)
        + "\n\n输出纯 JSON：\n"
        '{"shots":[{"clip_id":"片段id","cuts":["cand_x", ...]}]}\n'
        "cuts 按切分编号顺序排列，每个元素是对应切分选中的候选 id。"
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


def _segments_from_bounds(clip, bounds_s):
    timings = clip["char_timings"]
    segments = []
    for i in range(len(bounds_s) - 1):
        s_start, s_end = bounds_s[i], bounds_s[i + 1]
        chars = [c for c in timings
                 if s_start - 1e-6 <= (c["start"] + c["end"]) / 2 <= s_end + 1e-6]
        segments.append({
            "seg": i + 1,
            "start_ms": round(s_start * 1000),
            "end_ms": round(s_end * 1000),
            "narration": "".join(c["char"] for c in chars),
            "continuation": i > 0,
        })
    return segments


def _segments_legal(segments):
    for s in segments:
        dur = (s["end_ms"] - s["start_ms"]) / 1000.0
        if dur < SEEDANCE_MIN_S - 1e-6 or dur > SEEDANCE_MAX_S + 1e-6:
            return False
    return True


def _equal_split_segments(clip, n_hint):
    """Deterministic fallback: equal division snapped to nearest char boundary,
    bumping N until every segment is within [2s, 15s]."""
    start_s, end_s = _clip_bounds_s(clip)
    D = end_s - start_s
    snaps = _char_boundary_times(clip["char_timings"])
    segments = []
    for N in range(max(2, n_hint), max(2, n_hint) + 5):
        cut_times = []
        for k in range(1, N):
            ideal = start_s + k * D / N
            cut_times.append(
                min(snaps, key=lambda t: abs(t - ideal)) if snaps else ideal
            )
        cut_times = sorted(set(cut_times))
        segments = _segments_from_bounds(
            clip, [start_s] + cut_times + [end_s])
        if _segments_legal(segments):
            return segments
    return segments


def _realize_segments(clip, plan, picks_by_cut):
    start_s, end_s = _clip_bounds_s(clip)
    cut_times = []
    for cut in plan["cuts"]:
        if not cut["candidates"]:
            continue
        pick = picks_by_cut.get(cut["k"])
        cand = next((c for c in cut["candidates"] if c["id"] == pick), None)
        if cand is None:
            cand = cut["candidates"][len(cut["candidates"]) // 2]
        cut_times.append(cand["t"])
    cut_times = sorted(set(cut_times))
    segments = _segments_from_bounds(clip, [start_s] + cut_times + [end_s])
    if _segments_legal(segments):
        return segments
    print(f"  [{clip['clip_id']}] LLM cuts failed guard; using equal split",
          file=sys.stderr)
    return _equal_split_segments(clip, plan["N"])


def split_seedance_clips(clips, client, cfg):
    """Attach a segments[] array to every seedance clip.
    Long shots (>15s) are split; the LLM picks cut positions, the system
    enforces the [2s, 15s] guard. Short shots get a single segment.
    """
    plan_by_id = {}
    plans = []
    for clip in clips:
        if clip.get("method") != "seedance":
            continue
        if len(clip.get("char_timings", [])) < 2:
            continue
        if "start_ms" not in clip or "end_ms" not in clip:
            continue
        dur = (clip["end_ms"] - clip["start_ms"]) / 1000.0
        if dur > SEEDANCE_MAX_S:
            plan = _build_cut_plan(clip)
            plans.append(plan)
            plan_by_id[clip["clip_id"]] = plan

    picks = {}
    if plans:
        print(f"\nSplitting {len(plans)} long seedance shot(s); "
              f"LLM picks cut points...", file=sys.stderr)
        try:
            raw = stream_response(client, _build_split_message(plans), cfg)
            for entry in json.loads(raw).get("shots", []):
                cid = entry.get("clip_id")
                cut_ids = entry.get("cuts", [])
                picks[cid] = {j + 1: cut_ids[j]
                              for j in range(len(cut_ids))}
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"  Split LLM parse failed ({e}); equal split for all",
                  file=sys.stderr)

    split_count = 0
    for clip in clips:
        if clip.get("method") != "seedance":
            continue
        cid = clip.get("clip_id")
        if cid in plan_by_id:
            clip["segments"] = _realize_segments(
                clip, plan_by_id[cid], picks.get(cid, {}))
            split_count += 1
        else:
            clip["segments"] = _segments_from_bounds(
                clip, list(_clip_bounds_s(clip)))

    if plans:
        total = sum(len(c["segments"]) for c in clips
                    if c.get("method") == "seedance")
        print(f"  {split_count} shot(s) split; {total} total seedance segments.",
              file=sys.stderr)


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
        description="Stage 8: Clip plan → visual flow"
    )
    parser.add_argument(
        "--chapter", type=int, required=True,
        help="Chapter number (1-indexed).",
    )
    parser.add_argument(
        "--voice", choices=["female", "male"], default=None,
        help="Override voice (default: teacher's narration_gender).",
    )
    parser.add_argument(
        "--prompt", default="prompts/8_clip_plan_to_visual_flow.json",
        help="Path to visual flow prompt JSON.",
    )
    args = parser.parse_args()

    project_dir = Path(__file__).parent.parent
    load_dotenv(project_dir / ".env")
    args.voice = resolve_voice(project_dir, args.voice)

    chapter = args.chapter

    # ── Load inputs ────────────────────────────────────────────────────

    clip_plan_path = project_dir / "tmp" / "clip_plans" / f"clip_plan_chapter_{chapter}.json"
    if not clip_plan_path.exists():
        print(f"Error: clip plan not found: {clip_plan_path}", file=sys.stderr)
        print("Run stage 7 first.", file=sys.stderr)
        sys.exit(1)
    clip_plan = load_json(clip_plan_path)
    clips = clip_plan.get("clips", [])
    print(f"Loaded clip plan: {len(clips)} clips", file=sys.stderr)

    audio_dir = project_dir / "tmp" / "audio"
    lines_path = audio_dir / f"voice_chapter_{chapter}_{args.voice}_lines.json"
    alignment_path = audio_dir / f"voice_chapter_{chapter}_{args.voice}_alignment.json"

    if not lines_path.exists() or not alignment_path.exists():
        print(f"Error: audio files not found for chapter {chapter}", file=sys.stderr)
        print(f"  lines:     {lines_path}", file=sys.stderr)
        print(f"  alignment: {alignment_path}", file=sys.stderr)
        print("Run stage 6 first.", file=sys.stderr)
        sys.exit(1)

    lines = load_json(lines_path)
    alignment = load_json(alignment_path)
    print(
        f"Loaded {len(lines)} lines, {len(alignment['characters'])} aligned characters",
        file=sys.stderr,
    )

    # Group per-character timings by line, then by clip
    chars_by_line = group_chars_by_line(lines, alignment)
    clips = gather_clip_chars(clips, chars_by_line)

    # Load cast
    casts_dir = project_dir / "casts"
    cast_groups = gather_cast(casts_dir, chapter)
    total_cast = sum(len(v) for v in cast_groups.values())
    print(f"Gathered {total_cast} cast entries", file=sys.stderr)

    # Load art style
    art_style_path = project_dir / "prompts" / "art_style.json"
    art_style_text = "（未找到）"
    if art_style_path.exists():
        art_style = load_json(art_style_path)
        art_style_text = json.dumps(art_style, ensure_ascii=False, indent=2)

    # ── Load prompt ────────────────────────────────────────────────────

    prompt_path = project_dir / args.prompt
    if not prompt_path.exists():
        print(f"Error: prompt not found: {prompt_path}", file=sys.stderr)
        sys.exit(1)
    prompt = load_json(prompt_path)

    # ── Build LLM input ────────────────────────────────────────────────

    clips_text = format_clips_block(clips)
    cast_text = format_cast_block(cast_groups)

    system = build_system_prompt(prompt)
    user = build_user_message(prompt, clips_text, cast_text, art_style_text)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    # ── LLM call ───────────────────────────────────────────────────────

    api_key = os.environ.get(KIMI_CFG["api_key_env"], "")
    if not api_key:
        print(f"Error: {KIMI_CFG['api_key_env']} not set in .env", file=sys.stderr)
        sys.exit(1)
    api_base = os.environ.get(
        KIMI_CFG["api_base_env"], KIMI_CFG["api_base_default"]
    )
    client = OpenAI(api_key=api_key, base_url=api_base, timeout=600)

    print(
        f"\nCalling Kimi ({KIMI_CFG['model']}) for {len(clips)} clips...",
        file=sys.stderr,
    )
    t0 = time.time()
    raw = stream_response(client, messages, KIMI_CFG)
    elapsed = time.time() - t0

    # ── Parse output ───────────────────────────────────────────────────

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"Error: visual flow JSON parse failed: {e}", file=sys.stderr)
        debug_path = project_dir / "tmp" / "visual_flow" / f"visual_flow_chapter_{chapter}_raw.txt"
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(raw, encoding="utf-8")
        print(f"Raw output saved to {debug_path}", file=sys.stderr)
        sys.exit(1)

    design_clips = result.get("clips", [])
    designs_by_id = {d["clip_id"]: d for d in design_clips}

    # ── Validate ───────────────────────────────────────────────────────

    warnings = validate_designs(designs_by_id, clips)
    if warnings:
        print(f"\n⚠ 校验警告:", file=sys.stderr)
        for w in warnings:
            print(f"  - {w}", file=sys.stderr)

    # ── Merge designs into clip metadata ───────────────────────────────

    for clip in clips:
        cid = clip.get("clip_id")
        design = designs_by_id.get(cid, {})
        clip["visual_design"] = {
            k: v for k, v in design.items()
            if k not in ("clip_id", "method")  # already in clip metadata
        }
        # Promote char timings to output field (needed by step 9 for
        # frame-exact sync: subtitles, pointer highlighting, etc.)
        if "_char_timings" in clip:
            clip["char_timings"] = clip.pop("_char_timings")

    # Split long seedance shots into ≤15s segments (LLM picks cut points,
    # guard enforces [2s, 15s]). Per-segment prompts derived in stage 9.
    split_seedance_clips(clips, client, KIMI_CFG)

    method_counts = {}
    kind_counts = {}
    for clip in clips:
        m = clip.get("method", "?")
        method_counts[m] = method_counts.get(m, 0) + 1
        k = clip.get("visual_design", {}).get("kind", "?")
        kind_counts[k] = kind_counts.get(k, 0) + 1

    output = {
        "chapter": chapter,
        "chapter_name": clip_plan.get("chapter_name", f"Chapter {chapter}"),
        "model": KIMI_CFG["model"],
        "clip_count": len(clips),
        "elapsed_s": round(elapsed, 1),
        "method_counts": method_counts,
        "kind_counts": kind_counts,
        "clips": clips,
    }

    out_dir = project_dir / "tmp" / "visual_flow"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"visual_flow_chapter_{chapter}.json"
    save_json(output, out_path)

    method_str = ", ".join(f"{m}={n}" for m, n in sorted(method_counts.items()))
    kind_str = ", ".join(f"{k}={n}" for k, n in sorted(kind_counts.items()))
    print(
        f"\nSaved: {out_path} ({len(clips)} clips "
        f"[methods: {method_str}] [kinds: {kind_str}], {elapsed:.1f}s)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
