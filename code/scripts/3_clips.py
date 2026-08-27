#!/usr/bin/env python3
"""
Part 3: chapter → routed clips → timed visual plans → generated clip files.

Chapters run sequentially (API rate limits); clips within a chapter are
parallelized with a thread pool (describe + generate).

  route:    script + ElevenLabs timings + cast -> clips with method seedance|motion_graphics
  describe: routed clips -> clip-relative directives; seedance gets storyboard_prompt
  generate:
    seedance        -> GPT Image 2 storyboard -> OpenRouter Seedance video -> frame snap
    motion_graphics -> Kimi k3 writes #stage scene content -> 3_clips injects frozen
                       scripts/theme.ts -> deterministic Playwright 24fps render -> ffmpeg mp4

Inputs under projects/<project_id>/:
  script_final.json
  audio/voice_chapter_{N}_{voice}_lines.json
  audio/voice_chapter_{N}_{voice}_alignment.json
  casts/cast_set.json

Outputs under projects/<project_id>/:
  clip_plans/route_chapter_{N}.json
  clip_plans/visual_chapter_{N}.json
  clips/chapter_{N}/<clip_id>/clip.mp4
  clips/chapter_{N}/clips_manifest.json

Usage (run from code/ with venv activated):
    source .venv/bin/activate
    python scripts/3_clips.py --project-id triangles
    python scripts/3_clips.py --project-id triangles --chapter 1
"""

import argparse
import base64
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageOps


DEFAULT_MODEL = os.environ.get("KIMI_MODEL", "moonshotai/kimi-k3")
METHODS = {"seedance", "motion_graphics"}
GAP_TOLERANCE_MS = 60
FPS = 24
SEEDANCE_MIN_S = 4.0
SEEDANCE_MAX_S = 15.0
_SNIPPET_RADIUS = 10
OPENROUTER_API = "https://openrouter.ai/api/v1"
OPENROUTER_IMAGE_API = f"{OPENROUTER_API}/images"
OPENROUTER_VIDEO_API = f"{OPENROUTER_API}/videos"
IMAGE_MODEL = "openai/gpt-image-2"
SEEDANCE_MODEL = "bytedance/seedance-2.0"
MAX_WORKERS = 8  # concurrent clip-level describe/generate; chapters stay sequential
MOTION_ATTEMPTS = 3  # regenerate a rejected motion fragment, feeding errors back
SEEDANCE_ATTEMPTS = 3  # retry transient video failures; a rejected payload is fatal
SEEDANCE_POLL_TRIES = 90  # x10s = 15 min, above the observed tail for 720p 15s jobs
MIN_FRAGMENT_CHARS = 200


class StaticRenderError(RuntimeError):
    """The stage renders identically across the clip: blank or frozen output."""


class SeedanceFatal(RuntimeError):
    """A Seedance failure that retrying cannot fix (rejected payload)."""


class SeedanceJobDead(RuntimeError):
    """The provider reported the job failed/cancelled/expired: it cannot be collected."""

_TRACE_DIR = None
_TRACE_LOCK = threading.Lock()


def set_trace_dir(path):
    """Enable per-call LLM tracing to <path>/llm_trace.jsonl (thread-safe append)."""
    global _TRACE_DIR
    _TRACE_DIR = Path(path)
    _TRACE_DIR.mkdir(parents=True, exist_ok=True)


def _sanitize_messages(messages):
    """Copy messages with image payloads stripped (base64 data URLs are huge)."""
    cleaned = []
    for m in messages:
        m = dict(m)
        content = m.get("content")
        if isinstance(content, list):
            m["content"] = [
                {"type": "image_url", "image_url": "[omitted]"} if isinstance(p, dict) and p.get("type") == "image_url" else p
                for p in content
            ]
        cleaned.append(m)
    return cleaned


def trace_call(record):
    if _TRACE_DIR is None:
        return
    line = json.dumps({"ts": datetime.now(timezone.utc).isoformat(), **record}, ensure_ascii=False)
    with _TRACE_LOCK:
        with open(_TRACE_DIR / "llm_trace.jsonl", "a", encoding="utf-8") as f:
            f.write(line + "\n")
SEEDANCE_MAX_REFS = int(os.environ.get("SEEDANCE_MAX_REFS", "0"))  # 0 = no reference cap


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    """Write atomically: the clips manifest is rewritten as each clip settles, and a
    process killed mid-write must not leave a truncated file that 4_compose cannot parse.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def run_cmd(cmd, timeout=600):
    return subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=timeout, check=False)


def resolve_voice(code_dir, cli_voice):
    if cli_voice:
        return cli_voice
    try:
        pre = load_json(code_dir / "tools" / "prompts" / "1_preproduction.json")
        gender = pre.get("teacher_job", {}).get("narration_gender", "female")
        return gender if gender in ("male", "female") else "female"
    except Exception:  # noqa: BLE001
        return "female"


def chapter_from_script(script_obj, chapter_id):
    for i, ch in enumerate(script_obj.get("chapters", []) or [], 1):
        if isinstance(ch, dict) and ch.get("chapter_id", i) == chapter_id:
            return ch
    raise KeyError(f"chapter {chapter_id} not found")


def load_timed_lines(audio_dir, chapter_id, voice, script_chapter):
    path = audio_dir / f"voice_chapter_{chapter_id}_{voice}_lines.json"
    if not path.exists():
        raise FileNotFoundError(f"timed lines not found: {path}")
    raw = load_json(path)
    by_id = {item.get("line_id"): item for item in raw if isinstance(item, dict)}
    lines = []
    for line in script_chapter.get("lines", []) or []:
        if not isinstance(line, dict):
            continue
        lid = line.get("line_id")
        timed = by_id.get(lid)
        if not timed:
            raise ValueError(f"{path} missing timing for {lid}")
        text = (line.get("text") or "").strip()
        start_ms = timed.get("start_ms")
        end_ms = timed.get("end_ms")
        if start_ms is None and timed.get("start") is not None:
            start_ms = round(timed["start"] * 1000)
        if end_ms is None and timed.get("end") is not None:
            end_ms = round(timed["end"] * 1000)
        if start_ms is None or end_ms is None:
            raise ValueError(f"{lid} missing start/end in {path}")
        lines.append({"line_id": lid, "text": text, "start_ms": int(start_ms), "end_ms": int(end_ms)})
    if not lines:
        raise ValueError(f"chapter {chapter_id} has no lines")
    return lines


def load_chars_by_line(audio_dir, chapter_id, voice, timed_lines):
    path = audio_dir / f"voice_chapter_{chapter_id}_{voice}_alignment.json"
    if not path.exists():
        raise FileNotFoundError(f"alignment not found: {path}")
    alignment = load_json(path)
    chars = alignment.get("characters", [])
    starts = alignment.get("char_start_times", [])
    ends = alignment.get("char_end_times", [])
    by_line = {}
    idx = 0
    for line in timed_lines:
        n = len(line["text"])
        by_line[line["line_id"]] = [
            {"char": chars[j], "start_ms": round(starts[j] * 1000), "end_ms": round(ends[j] * 1000)}
            for j in range(idx, min(idx + n, len(chars)))
        ]
        idx += n
    if idx != len(chars):
        print(f"  warning: consumed {idx}/{len(chars)} alignment chars", file=sys.stderr)
    return by_line


def load_cast_set(project_dir):
    path = project_dir / "casts" / "cast_set.json"
    if not path.exists():
        return {"cast": []}
    return load_json(path)


def available_cast(cast_set, chapter_id):
    out = []
    for entry in cast_set.get("cast", []) or []:
        if not isinstance(entry, dict):
            continue
        if entry.get("is_anchor"):
            out.append(entry)
            continue
        chapters = entry.get("appears_in_chapters")
        if not isinstance(chapters, list) or chapter_id in chapters:
            out.append(entry)
    return out


def format_timed_script(timed_lines, chars_by_line):
    blocks = []
    for line in timed_lines:
        lid = line["line_id"]
        blocks.append(f"{lid} [{line['start_ms']}ms-{line['end_ms']}ms] {line['text']}")
        chars = chars_by_line.get(lid, [])
        if chars:
            blocks.append("  chars: " + " ".join(f"{c['char']}({c['start_ms']}-{c['end_ms']})" for c in chars))
    return "\n".join(blocks)


def format_cast(entries):
    if not entries:
        return "（本章无可用 cast）"
    return "\n".join(
        f"[{e.get('cast_id', '?')}] {e.get('name', '?')}（{e.get('kind', '?')}）: {e.get('description', '')}"
        for e in entries
    )


def image_to_data_url(path):
    ext = Path(path).suffix.lower().lstrip(".")
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext or 'png'}"
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode("utf-8")


def save_jpg(img, out_path, max_edge=1024):
    """Save a storyboard frame as exact 1280x720, 100 DPI.

    Unlike the cast references in 2_casts.py, cropping to 16:9 is correct here: a
    storyboard is submitted to Seedance as first_frame for a 16:9 clip, so it has to
    match the video's geometry rather than preserve its own.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = ImageOps.fit(img.convert("RGB"), (1280, 720), Image.LANCZOS, centering=(0.5, 0.5))
    img.save(out_path, format="JPEG", quality=90, dpi=(100, 100))


def resolve_code_path(code_dir, rel):
    return (code_dir / rel).resolve()


def cast_reference_images(code_dir, cast_entries, clip_cast_ids, limit=None):
    """All reference images for the cast/location entries needed by this clip.
    No artificial count limit: characters and environments may both be needed."""
    by_id = {e.get("cast_id"): e for e in cast_entries}
    paths = []
    for cid in clip_cast_ids or []:
        entry = by_id.get(cid)
        if not entry:
            continue
        for ref in entry.get("reference_images", []) or []:
            p = resolve_code_path(code_dir, ref)
            if p.exists():
                paths.append(p)
            if limit is not None and len(paths) >= limit:
                return paths
    return paths


def call_image_api(api_key, prompt, input_references=None, label=""):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": IMAGE_MODEL, "prompt": prompt, "resolution": "1K", "quality": "high", "output_format": "png"}
    if input_references:
        payload["input_references"] = input_references
    for attempt in (1, 2):
        try:
            resp = requests.post(OPENROUTER_IMAGE_API, headers=headers, json=payload, timeout=300)
            resp.raise_for_status()
            result = resp.json()
            if not result.get("data"):
                raise RuntimeError(f"no image returned: {result}")
            trace_call({"kind": "image_call", "label": label, "model": IMAGE_MODEL, "prompt": prompt, "n_refs": len(input_references or [])})
            return base64.b64decode(result["data"][0]["b64_json"])
        except Exception as e:  # noqa: BLE001
            print(f"    image attempt {attempt}/2 failed: {e}", file=sys.stderr)
            if attempt == 1:
                time.sleep(3)
                continue
            raise


def _seedance_ref_rank(path):
    """Explicit reference priority when the API forces a cap.
    Lower rank survives: storyboard > location/classroom > teacher > students > other cast.
    Within the same subject, prefer design over example/wiki.
    """
    name = Path(path).name.lower()
    if "storyboard" in name:
        group = 0
    elif "location" in name or "classroom" in name:
        group = 1
    elif "teacher" in name:
        group = 2
    elif "xiaoming" in name or "xiaohong" in name or "student" in name:
        group = 3
    else:
        group = 4
    sub = 0 if "design" in name else (2 if "wiki" in name else 1)
    return group, sub, name


def select_seedance_refs(reference_paths):
    paths = []
    seen = set()
    for p in reference_paths or []:
        rp = str(Path(p).resolve())
        if rp not in seen and Path(p).exists():
            seen.add(rp)
            paths.append(p)
    ordered = sorted(paths, key=_seedance_ref_rank)
    if SEEDANCE_MAX_REFS > 0:
        return ordered[:SEEDANCE_MAX_REFS]
    return ordered


def _seedance_submit(api_key, prompt, duration_s, out_path, reference_paths=None, first_frame_path=None, prev_last_frame_b64=None, label=""):
    """Submit one segment and return its job id. Everything after this point is billed."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": SEEDANCE_MODEL,
        "prompt": prompt,
        "duration": max(4, min(math.ceil(duration_s), 15)),
        "resolution": "720p",
        "aspect_ratio": "16:9",
        "generate_audio": False,
    }
    selected_refs = select_seedance_refs(reference_paths or [])
    if len(selected_refs) < len(reference_paths or []):
        used = {str(Path(p).resolve()) for p in selected_refs}
        dropped = [Path(p).name for p in (reference_paths or []) if str(Path(p).resolve()) not in used]
        print(f"  seedance refs capped at {SEEDANCE_MAX_REFS}; dropped: {dropped}", file=sys.stderr)
    refs = []
    for p in selected_refs:
        refs.append({"type": "image_url", "image_url": {"url": image_to_data_url(p)}})
    if refs:
        payload["input_references"] = refs
    if prev_last_frame_b64:
        payload["frame_images"] = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{prev_last_frame_b64}"}, "frame_type": "first_frame"}
        ]
    elif first_frame_path and Path(first_frame_path).exists():
        payload["frame_images"] = [
            {"type": "image_url", "image_url": {"url": image_to_data_url(first_frame_path)}, "frame_type": "first_frame"}
        ]

    resp = requests.post(OPENROUTER_VIDEO_API, headers=headers, json=payload, timeout=60)
    if resp.status_code not in (200, 202):
        msg = f"seedance submit failed {resp.status_code}: {resp.text[:500]}"
        # A 4xx means the payload itself was rejected (too many references, bad prompt);
        # resending it unchanged just spends money to fail identically. 429 is transient.
        if 400 <= resp.status_code < 500 and resp.status_code != 429:
            raise SeedanceFatal(msg)
        raise RuntimeError(msg)
    job_id = resp.json().get("id")
    if not job_id:
        raise RuntimeError(f"seedance submit returned no id: {resp.text[:500]}")
    trace_call(
        {
            "kind": "video_call",
            "label": label,
            "model": SEEDANCE_MODEL,
            "job_id": job_id,
            "prompt": prompt,
            "duration_s": duration_s,
            "n_refs": len(refs),
            "has_first_frame": bool(payload.get("frame_images")),
        }
    )

    return job_id


def _seedance_collect(api_key, job_id, out_path):
    """Poll one already-submitted job to completion and download it.

    Deliberately separate from submission: everything here happens AFTER the generation
    has been bought, so a failure in this half must be retried against the same job_id
    rather than by buying another one.
    """
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for _ in range(SEEDANCE_POLL_TRIES):
        time.sleep(10)
        poll = requests.get(f"{OPENROUTER_VIDEO_API}/{job_id}", headers=headers, timeout=30)
        poll.raise_for_status()
        data = poll.json()
        status = data.get("status")
        if status == "completed":
            urls = data.get("unsigned_urls") or []
            if not urls:
                raise RuntimeError("seedance completed but no unsigned_urls")
            video = requests.get(urls[0], headers=headers, timeout=120)
            video.raise_for_status()
            body = video.content
            if len(body) < 1024 or body[4:8] != b"ftyp":
                raise RuntimeError("downloaded seedance result is not mp4")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(body)
            return
        if status in ("failed", "cancelled", "expired"):
            raise SeedanceJobDead(f"seedance job {status}: {str(data)[:300]}")
    raise RuntimeError("seedance polling timed out")


def submit_seedance(api_key, prompt, duration_s, out_path, reference_paths=None, first_frame_path=None, prev_last_frame_b64=None, label=""):
    """Generate one Seedance segment, retrying transient failures without re-buying.

    Video generation is the most expensive step and the most likely to fail for reasons
    that clear on their own. Without any retry a single blip cost the whole chapter,
    since the exception unwinds through the clip pool. But a retry that re-POSTs is
    almost as bad: a hiccup while polling or downloading would abandon a generation
    already paid for and order another. So the job id is held across attempts and only
    a job the provider reports as dead -- or a submit that never produced an id --
    causes a new submission.
    """
    job_id = None
    last = None
    for attempt in range(1, SEEDANCE_ATTEMPTS + 1):
        try:
            if job_id is None:
                job_id = _seedance_submit(
                    api_key, prompt, duration_s, out_path,
                    reference_paths=reference_paths,
                    first_frame_path=first_frame_path,
                    prev_last_frame_b64=prev_last_frame_b64,
                    label=label,
                )
            return _seedance_collect(api_key, job_id, out_path)
        except SeedanceFatal:
            raise
        except SeedanceJobDead as e:
            last = e
            job_id = None  # the generation itself failed; finishing it is not possible
            print(f"    seedance attempt {attempt}/{SEEDANCE_ATTEMPTS}: {e}", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            last = e
            where = "submit" if job_id is None else f"poll/download of job {job_id}"
            print(f"    seedance attempt {attempt}/{SEEDANCE_ATTEMPTS} failed at {where}: {e}", file=sys.stderr)
        if attempt < SEEDANCE_ATTEMPTS:
            # Jitter by clip so 8 concurrent workers do not retry in lockstep.
            time.sleep(15 * attempt + (hash(label) % 7))
    raise RuntimeError(f"seedance failed after {SEEDANCE_ATTEMPTS} attempts: {last}")


def frame_snap(input_path, target_frames, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)
    tmp = output_path.with_suffix(".snap.mp4")
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-vf", "scale=1280:720,tpad=stop_mode=clone:stop_duration=60",
        "-frames:v", str(max(1, int(target_frames))),
        "-r", str(FPS), "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
        str(tmp),
    ]
    proc = run_cmd(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg frame_snap failed: {proc.stderr[-500:]}")
    tmp.replace(output_path)


def encode_frames_to_mp4(frames_dir, out_path, input_fps, total_frames=None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-framerate", str(input_fps),
        "-i", str(frames_dir / "frame_%04d.png"),
        "-vf", "scale=1280:720", "-r", str(FPS),
    ]
    # Cap the output so the clip can never exceed the frames the plan allotted it,
    # whatever happens to be sitting in frames_dir.
    if total_frames:
        cmd += ["-frames:v", str(int(total_frames))]
    cmd += [
        "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    proc = run_cmd(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg encode frames failed: {proc.stderr[-500:]}")


def stream_json(client, messages, model, label=""):
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=100000,
        stream=True,
        response_format={"type": "json_object"},
    )
    thinking = False
    thinking_parts = []
    parts = []
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
            thinking_parts.append(reasoning)
        if delta.content:
            if thinking:
                thinking = False
                print("\n--- answer ---", file=sys.stderr)
            print(delta.content, end="", flush=True)
            parts.append(delta.content)
    print(file=sys.stderr)
    content = "".join(parts)
    trace_call(
        {
            "kind": "llm_call",
            "label": label,
            "model": model,
            "messages": _sanitize_messages(messages),
            "reasoning": "".join(thinking_parts),
            "output": content,
        }
    )
    return content


def call_llm_json(client, messages, model, validate_fn, label):
    last_errors = ["not called"]
    for attempt in range(1, 3):
        raw = stream_json(client, messages, model, label=f"{label} attempt {attempt}")
        messages.append({"role": "assistant", "content": raw})
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            parsed = None
            last_errors = [f"JSON parse failed: {e}"]
        if parsed is not None:
            last_errors = validate_fn(parsed)
            if not last_errors:
                return parsed
        messages.append(
            {
                "role": "user",
                "content": f"{label} validation failed. Return corrected pure JSON only.\n"
                + json.dumps({"errors": last_errors}, ensure_ascii=False, indent=2),
            }
        )
    raise RuntimeError(f"{label} failed after retries: {last_errors}")


def enrich_clip_times(clips, lines_by_id):
    for clip in clips:
        starts, ends = [], []
        for lid in clip.get("line_ids", []) or []:
            line = lines_by_id.get(lid)
            if line:
                starts.append(line["start_ms"])
                ends.append(line["end_ms"])
        if starts and ends:
            clip["start_ms"] = min(starts)
            clip["end_ms"] = max(ends)
            clip["duration_ms"] = clip["end_ms"] - clip["start_ms"]
    return clips


def snap_clips_to_frames(clips):
    """Give every clip an integer frame count whose boundaries telescope.

    Clips are placed by concatenation, not by seeking to start_ms, so rounding each
    duration to whole frames independently lets the video drift against the narration
    and the drift compounds with clip count. Chaining each clip's start frame to the
    previous clip's end frame means the total is always round(chapter_end_ms) frames
    and the error can never exceed half a frame no matter how many clips there are.

    Idempotent: end_frame derives from end_ms, which this never modifies.
    """
    prev_end_frame = 0
    for clip in clips:
        if clip.get("start_ms") is None or clip.get("end_ms") is None:
            continue
        end_frame = max(prev_end_frame + 1, round(clip["end_ms"] / 1000 * FPS))
        clip["start_frame"] = prev_end_frame
        clip["end_frame"] = end_frame
        clip["frames"] = end_frame - prev_end_frame
        clip["duration_ms"] = round(clip["frames"] * 1000 / FPS)
        prev_end_frame = end_frame
    return clips


def _video_is_static(path, duration_ms):
    """Do three frames sampled across the video come out byte-identical?

    Encoded-file counterpart of the pre-flight probe in render_motion_html. It exists
    because blank clips predate the fragment validator: a 21s file of flat background
    has exactly the right geometry and frame count, so without this the resume path
    would happily adopt one and stamp it ok. A probe that cannot run counts as static,
    since regenerating a good clip only costs time, while reusing a blank one ships it.
    """
    digests = set()
    for fraction in (0.1, 0.5, 0.9):
        try:
            proc = subprocess.run(
                ["ffmpeg", "-v", "error", "-ss", f"{duration_ms / 1000 * fraction:.3f}",
                 "-i", str(path), "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "pipe:1"],
                capture_output=True, timeout=60,
            )
        except Exception:  # noqa: BLE001
            return True
        if proc.returncode != 0 or not proc.stdout:
            return True
        digests.add(hashlib.md5(proc.stdout).hexdigest())
    return len(digests) == 1


def _clip_output_ok(path, duration_ms, expected_frames=None):
    """Is an existing clip/segment mp4 trustworthy enough to skip regenerating it?

    Guards the resume path. A file on disk may be truncated, may be from a stale plan,
    or may be a blank render from before the fragment validator existed. Checks that the
    geometry and frame count match what the current plan asks for, and that the picture
    actually changes.
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size < 1024:
        return False
    want = int(expected_frames or max(1, round(duration_ms / 1000 * FPS)))
    try:
        proc = run_cmd(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,nb_frames", "-of", "json", str(path)],
            timeout=60,
        )
        if proc.returncode != 0:
            return False
        stream = (json.loads(proc.stdout or "{}").get("streams") or [{}])[0]
        if stream.get("width") != 1280 or stream.get("height") != 720:
            return False
        if abs(int(stream.get("nb_frames") or 0) - want) > 1:
            return False
    except Exception:  # noqa: BLE001
        return False
    return not _video_is_static(path, duration_ms)


def validate_route_factory(timed_lines, cast_ids):
    lines_by_id = {l["line_id"]: l for l in timed_lines}
    expected = [l["line_id"] for l in timed_lines]
    pos = {lid: i for i, lid in enumerate(expected)}

    def validate(parsed):
        errors = []
        clips = parsed.get("clips") if isinstance(parsed, dict) else None
        if not isinstance(clips, list) or not clips:
            return ["route output must have non-empty clips"]
        enrich_clip_times(clips, lines_by_id)
        flat = []
        seen = set()
        for clip in clips:
            if not isinstance(clip, dict):
                errors.append("clip must be object")
                continue
            cid = clip.get("clip_id")
            if not cid or cid in seen:
                errors.append(f"bad or duplicate clip_id {cid}")
            seen.add(cid)
            lids = clip.get("line_ids")
            if not isinstance(lids, list) or not lids:
                errors.append(f"{cid}: line_ids must be non-empty")
                continue
            idx = [pos[x] for x in lids if x in pos]
            if len(idx) != len(lids):
                errors.append(f"{cid}: unknown line_id in {lids}")
            elif idx != list(range(idx[0], idx[0] + len(idx))):
                errors.append(f"{cid}: line_ids not contiguous")
            flat.extend(lids)
            if clip.get("method") not in METHODS:
                errors.append(f"{cid}: method must be seedance or motion_graphics")
            if not clip.get("summary"):
                errors.append(f"{cid}: missing summary")
            for cast_id in clip.get("cast_ids", []) or []:
                if cast_id not in cast_ids:
                    errors.append(f"{cid}: unknown cast_id {cast_id}")
        if flat != expected:
            errors.append("route clips must cover all lines in order with no overlap")
        return errors
    return validate


def _valid_rel(value):
    return isinstance(value, list) and len(value) == 2 and all(isinstance(x, (int, float)) for x in value) and value[0] < value[1]


def _norm_windows(items, duration, gapless, exactness, label):
    errors = []
    if not isinstance(items, list) or not items:
        return [f"{label} must be a non-empty list"], []
    out = []
    for i, d in enumerate(items):
        if not isinstance(d, dict) or not _valid_rel(d.get("t_ms")) or not d.get("action"):
            errors.append(f"{label} {i} needs t_ms [start,end] and action")
            continue
        t = d["t_ms"]
        if t[0] < -GAP_TOLERANCE_MS or t[1] > duration + GAP_TOLERANCE_MS:
            errors.append(f"{label} {i} outside [0,{duration}]")
        out.append({"t_rel_ms": [round(t[0]), round(t[1])], "action": d["action"], "exactness": exactness})
    ts = [d["t_rel_ms"] for d in out]
    for i in range(len(ts) - 1):
        gap = ts[i + 1][0] - ts[i][1]
        if gapless:
            if abs(gap) > GAP_TOLERANCE_MS:
                errors.append(f"{label} gap/overlap {gap}ms between {ts[i]} and {ts[i + 1]}")
        elif gap < 0:
            errors.append(f"{label} overlap {gap}ms between {ts[i]} and {ts[i + 1]}")
    if gapless and ts:
        if abs(ts[0][0] - 0) > GAP_TOLERANCE_MS:
            errors.append(f"first {label} must start at 0")
        if abs(ts[-1][1] - duration) > GAP_TOLERANCE_MS:
            errors.append(f"last {label} must end at {duration}")
    return errors, out


def attach_clip_char_timings(clips, chars_by_line):
    for clip in clips:
        timings = []
        for lid in clip.get("line_ids", []) or []:
            timings.extend(chars_by_line.get(lid, []))
        clip["_char_timings"] = timings
    return clips


def _clip_bounds_s(clip):
    return clip["start_ms"] / 1000.0, clip["end_ms"] / 1000.0


def _text_of(timings):
    return "".join(c["char"] for c in timings)


def _char_boundary_times(timings):
    return sorted({round(c["end_ms"] / 1000.0, 3) for c in timings})


def _build_cut_plan(clip):
    timings = clip["_char_timings"]
    text = _text_of(timings)
    start_s, end_s = _clip_bounds_s(clip)
    D = end_s - start_s
    N = max(2, math.ceil(D / SEEDANCE_MAX_S))
    idx_bounds = [(i, c["end_ms"] / 1000.0) for i, c in enumerate(timings) if i + 1 < len(timings)]

    cuts = []
    for k in range(1, N):
        ideal = start_s + k * D / N
        W = min(D / N * 0.4, 3.0)
        lo = max(start_s + SEEDANCE_MIN_S, ideal - W)
        hi = min(end_s - SEEDANCE_MIN_S, ideal + W)
        in_win = sorted(((ci, t) for ci, t in idx_bounds if lo <= t <= hi), key=lambda x: x[1])
        if len(in_win) > 6:
            step = len(in_win) / 6.0
            in_win = [in_win[min(len(in_win) - 1, int(i * step))] for i in range(6)]
        candidates = []
        for j, (ci, t) in enumerate(in_win):
            before = text[max(0, ci - _SNIPPET_RADIUS + 1):ci + 1]
            after = text[ci + 1:ci + 1 + _SNIPPET_RADIUS]
            candidates.append({"id": f"cand_{chr(ord('a') + j)}", "t": round(t, 2), "before": before, "after": after})
        cuts.append({"k": k, "ideal": round(ideal, 2), "lo": round(lo, 2), "hi": round(hi, 2), "candidates": candidates})
    return {"clip_id": clip["clip_id"], "N": N, "cuts": cuts}


def _build_split_message(plans):
    system = (
        "你的任务是为视频片段的旁白选择最自然的语义切分点。"
        "这些片段由 seedance 生成，seedance 有一条硬性限制：切分后每一段时长都必须在 4 到 15 秒之间。"
        "候选位置都已保证落在合法区间内，你只需挑最自然的。优先句末（。！？）或分句末（，；、），"
        "让每段旁白尽量是一个完整语意单元。只能从候选中选择，返回候选 id。"
    )
    blocks = []
    for plan in plans:
        lines = [f"片段 {plan['clip_id']}，共需 {len(plan['cuts'])} 处切分："]
        for cut in plan["cuts"]:
            if not cut["candidates"]:
                continue
            lines.append(f"  切分 {cut['k']}（理想 {cut['ideal']}s，范围 {cut['lo']}-{cut['hi']}s）：")
            for cand in cut["candidates"]:
                lines.append(f"    {cand['id']}（{cand['t']}s）：「…{cand['before']}」|「{cand['after']}…」")
        blocks.append("\n".join(lines))
    user = "\n\n".join(blocks) + "\n\n输出纯 JSON：\n{\"shots\":[{\"clip_id\":\"片段id\",\"cuts\":[\"cand_x\", ...]}]}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _segments_from_bounds(clip, bounds_s):
    timings = clip["_char_timings"]
    segments = []
    for i in range(len(bounds_s) - 1):
        s_start, s_end = bounds_s[i], bounds_s[i + 1]
        chars = [c for c in timings if s_start - 1e-6 <= (c["start_ms"] + c["end_ms"]) / 2000.0 <= s_end + 1e-6]
        segments.append(
            {
                "seg": i + 1,
                "start_ms": round(s_start * 1000),
                "end_ms": round(s_end * 1000),
                "narration": "".join(c["char"] for c in chars),
                "continuation": i > 0,
            }
        )
    return segments


def _segments_legal(segments):
    for s in segments:
        dur = (s["end_ms"] - s["start_ms"]) / 1000.0
        if dur < SEEDANCE_MIN_S - 1e-6 or dur > SEEDANCE_MAX_S + 1e-6:
            return False
    return True


def _equal_split_segments(clip, n_hint):
    start_s, end_s = _clip_bounds_s(clip)
    D = end_s - start_s
    snaps = _char_boundary_times(clip["_char_timings"])
    segments = []
    for N in range(max(2, n_hint), max(2, n_hint) + 5):
        cut_times = []
        for k in range(1, N):
            ideal = start_s + k * D / N
            cut_times.append(min(snaps, key=lambda t: abs(t - ideal)) if snaps else ideal)
        cut_times = sorted(set(cut_times))
        segments = _segments_from_bounds(clip, [start_s] + cut_times + [end_s])
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
    print(f"  [{clip['clip_id']}] LLM cuts failed guard; using equal split", file=sys.stderr)
    return _equal_split_segments(clip, plan["N"])


def split_seedance_clips(clips, client, model):
    plan_by_id = {}
    plans = []
    for clip in clips:
        if clip.get("method") != "seedance" or clip.get("segments"):
            continue  # already split on a previous run; the plan is persisted
        if len(clip.get("_char_timings", [])) < 2 or "start_ms" not in clip or "end_ms" not in clip:
            continue
        dur = (clip["end_ms"] - clip["start_ms"]) / 1000.0
        if dur > SEEDANCE_MAX_S:
            plan = _build_cut_plan(clip)
            plans.append(plan)
            plan_by_id[clip["clip_id"]] = plan

    picks = {}
    if plans:
        print(f"Splitting {len(plans)} long seedance shot(s); LLM picks cut points...", file=sys.stderr)
        try:
            raw = stream_json(client, _build_split_message(plans), model, label="split_seedance")
            for entry in json.loads(raw).get("shots", []):
                cid = entry.get("clip_id")
                cut_ids = entry.get("cuts", [])
                picks[cid] = {j + 1: cut_ids[j] for j in range(len(cut_ids))}
        except Exception as e:  # noqa: BLE001
            print(f"  Split LLM failed ({e}); equal split for all", file=sys.stderr)

    for clip in clips:
        if clip.get("method") != "seedance" or clip.get("segments"):
            continue
        cid = clip["clip_id"]
        if cid in plan_by_id:
            clip["segments"] = _realize_segments(clip, plan_by_id[cid], picks.get(cid, {}))
        else:
            clip["segments"] = _segments_from_bounds(clip, list(_clip_bounds_s(clip)))
    return clips


def validate_describe_for_clip(clip, parsed):
    duration = clip["duration_ms"]
    if clip["method"] == "seedance":
        errors = []
        if not isinstance(parsed.get("storyboard_plan"), list) or not parsed["storyboard_plan"]:
            errors.append("seedance describe needs non-empty storyboard_plan")
        w_errors, directives = _norm_windows(parsed.get("directives"), duration, False, "approximate", "seedance directive")
        errors.extend(w_errors)
        if not errors:
            clip["describe"] = {"storyboard_plan": parsed["storyboard_plan"], "directives": directives}
        return errors

    errors, keyframes = _norm_windows(parsed.get("keyframe_plan"), duration, True, "exact", "motion keyframe")
    if not errors:
        clip["describe"] = {"keyframe_plan": keyframes}
    return errors


def build_route_messages(prompt, timed_script, cast_text):
    cfg = prompt["route"]
    constraints = "\n".join(f"- {c}" for c in cfg.get("constraints", []))
    system = f"{cfg['routing']}\n\n{cfg['output_format']}\n\n约束：\n{constraints}"
    user = cfg["task"].replace("{timed_script}", timed_script).replace("{cast_set}", cast_text)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_describe_messages(prompt, clip, chapter_context, cast_text):
    cfg = prompt["seedance_describe"] if clip["method"] == "seedance" else prompt["motion_describe"]
    constraints = "\n".join(f"- {c}" for c in cfg.get("constraints", []))
    system = f"{cfg['rules']}\n\n{cfg['output_format']}\n\n约束：\n{constraints}"
    user = (
        "本章其他 clip（用于整章连贯，不要重复设计）：\n"
        f"{chapter_context}\n\n"
        "当前要详细设计的 clip：\n"
        f"{json.dumps(clip, ensure_ascii=False, indent=2)}\n\n"
        f"本 clip 可用 cast：\n{cast_text}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def directives_text(clip):
    parts = []
    for d in clip.get("describe", {}).get("directives", []) or []:
        parts.append(f"{d.get('t_rel_ms')}: {d.get('action')}")
    return "\n".join(parts)


def stream_text(client, messages, model, label=""):
    stream = client.chat.completions.create(model=model, messages=messages, max_tokens=100000, stream=True)
    thinking_parts = []
    parts = []
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            thinking_parts.append(reasoning)
        if delta.content:
            print(delta.content, end="", flush=True)
            parts.append(delta.content)
    print(file=sys.stderr)
    content = "".join(parts)
    trace_call(
        {
            "kind": "llm_call",
            "label": label,
            "model": model,
            "messages": _sanitize_messages(messages),
            "reasoning": "".join(thinking_parts),
            "output": content,
        }
    )
    return content


def _strip_fence(text):
    text = text.strip()
    text = re.sub(r"^```(?:html)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def build_motion_messages(cfg, clip, theme_css):
    """Opening turns for a motion fragment.

    The frozen theme is pasted into the system message. It was previously only injected
    into the rendered HTML, so the model never saw the palette or the edu-* classes it
    was told not to redefine -- and predictably ignored them, hard-coding its own hexes.
    """
    constraints = "\n".join(f"- {c}" for c in cfg.get("constraints", []))
    system = (
        f"{cfg['html_rules']}\n\n约束：\n{constraints}\n\n"
        "以下冻结主题 CSS 已经注入到页面，是颜色、字体、字号、间距的唯一来源。"
        "直接使用其中的 CSS 变量和 edu-* 类（SVG 用 fill: currentColor 继承，"
        "所以 .edu-primary 这类类同样作用于 SVG 图形）；不要另外写死颜色或字体：\n"
        f"{theme_css}"
    )
    user = (
        cfg["task"]
        .replace("{summary}", clip.get("summary", ""))
        .replace("{duration_ms}", str(clip["duration_ms"]))
        .replace("{keyframe_plan}", json.dumps(clip.get("describe", {}).get("keyframe_plan", []), ensure_ascii=False, indent=2))
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def load_theme_css(code_dir):
    path = code_dir / "scripts" / "theme.ts"
    text = path.read_text(encoding="utf-8")
    m = re.search(r"export const THEME_CSS = `([\s\S]*)`;", text)
    if not m:
        raise RuntimeError(f"THEME_CSS not found in {path}")
    return m.group(1)


_ELEMENT_RE = re.compile(
    r"<(div|span|svg|p|h[1-6]|section|ul|ol|li|table|canvas|g|rect|circle|path|text|line|polygon|polyline|ellipse)\b",
    re.I,
)
# Any http(s) URL except the W3C namespace URIs. createElementNS REQUIRES
# "http://www.w3.org/2000/svg", so a blanket "http://" ban rejects every fragment that
# builds SVG imperatively -- which is exactly the window.renderAt pattern that renders
# best. That ban destroyed a valid 10k-char fragment (c2_clip_004) and with it a whole
# chapter's manifest, discarding three already-rendered sibling clips.
_EXTERNAL_URL_RE = re.compile(r"https?://(?!www\.w3\.org/)", re.I)

def _has_external_url_func(fragment):
    """Does any url(...) point outside this fragment?

    url(#id) targets a marker/gradient/clipPath defined in the same fragment and is
    safe; url(foo.png) or url("http://...") fetches a resource and is not. Written as a
    scan rather than one regex because the quoting is variable -- url(#g), url('#g'),
    url( "#g" ) are all legal CSS/SVG -- and a lookahead after optional quotes silently
    backtracks past them, which would reject exactly the internal references the prompt
    tells the model to use.
    """
    for m in re.finditer(r"url\(", fragment, re.I):
        rest = fragment[m.end():].lstrip().lstrip("\"'").lstrip()
        if not rest.startswith("#"):
            return True
    return False


def validate_motion_fragment(fragment):
    """Post-conditions on the #stage fragment, both negative and positive.

    Negative: no document scaffolding, no external resources, no media elements.
    Positive: non-empty, long enough to be a scene, containing something renderable.

    The positive half is what stops the silent failure: an empty fragment wraps into a
    legal HTML document, renders as N frames of flat background, and gets stamped ok.
    """
    text = (fragment or "").strip()
    if not text:
        return ["fragment is empty: the model returned nothing"]

    forbidden = ["<html", "<head", "<body", "<style", "<link", "@import", "<audio", "<video", "<img", "<iframe"]
    errors = [f"forbidden fragment content: {x}" for x in forbidden if x in fragment]
    if _EXTERNAL_URL_RE.search(fragment):
        errors.append("forbidden external URL (the http://www.w3.org/... namespace URIs are allowed)")
    if _has_external_url_func(fragment):
        errors.append("forbidden external url() reference (internal url(#id) is allowed)")
    if len(text) < MIN_FRAGMENT_CHARS:
        errors.append(f"fragment is only {len(text)} chars, too short to be a real scene")
    if not _ELEMENT_RE.search(text):
        errors.append("fragment contains no renderable element")
    return errors


def wrap_motion_html(fragment, duration_ms, theme_css):
    return (
        "<!doctype html><html><head><meta charset='utf-8'><style>"
        + theme_css
        + "</style></head><body><div id='stage'>"
        + fragment
        + "</div><script>window.__DURATION_MS="
        + str(int(duration_ms))
        + ";</script></body></html>"
    )


def render_motion_html(html_path, out_path, duration_ms, total_frames=None):
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("motion renderer needs Python Playwright + Chromium installed") from e

    frames_dir = Path(html_path).parent / "frames_render"
    total_frames = max(1, int(total_frames or round(duration_ms / 1000 * FPS)))
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 720})
        page.goto("file://" + str(Path(html_path).resolve()))
        page.evaluate("document.getAnimations().forEach(a=>{a.pause(); a.currentTime=0;})")

        def seek(t):
            page.evaluate(
                "(t)=>{ if (window.renderAt) window.renderAt(t); document.getAnimations().forEach(a=>{a.pause(); a.currentTime=t;}); }",
                t,
            )

        # Pre-flight. A stage that is byte-identical at five points across its span is
        # blank or frozen; catch it in ~2s rather than after a 500-frame render. Runs
        # before frames_dir is created so a rejected clip leaves nothing behind.
        probes = set()
        for fraction in (0.0, 0.25, 0.5, 0.75, 0.98):
            seek(round(duration_ms * fraction))
            probes.add(hashlib.md5(page.screenshot()).hexdigest())
        if len(probes) == 1:
            browser.close()
            raise StaticRenderError(
                "stage is identical at 0/25/50/75/98% of the clip: nothing renders, or nothing moves"
            )

        # Clear frames from any earlier attempt first. encode_frames_to_mp4 ingests the
        # whole contiguous frame_%04d.png run, so a stale trailing frame from a longer
        # previous render would be appended to this clip -- both making it a frame too
        # long and ending it on leftover picture from the render being replaced.
        shutil.rmtree(frames_dir, ignore_errors=True)
        frames_dir.mkdir(parents=True, exist_ok=True)
        for i in range(total_frames):
            seek(round(i / FPS * 1000))
            page.screenshot(path=str(frames_dir / f"frame_{i:04d}.png"))
        browser.close()
    encode_frames_to_mp4(frames_dir, out_path, FPS, total_frames)
    # Only on success: a failed render keeps its frames for inspection.
    shutil.rmtree(frames_dir, ignore_errors=True)


def generate_motion_clip(client, model, prompt, clip, clip_dir):
    """Generate -> validate -> render, retrying with the rejection reason fed back.

    Mirrors call_llm_json's contract: the model gets told exactly what was wrong and
    tries again. A single-shot call here is how an empty response became a blank clip.
    Infrastructure failures (no Playwright) raise straight out rather than burning
    attempts on an error the model cannot fix.
    """
    clip_dir.mkdir(parents=True, exist_ok=True)
    theme_css = load_theme_css(Path(__file__).parent.parent)
    cfg = prompt["motion_generate"]
    messages = build_motion_messages(cfg, clip, theme_css)
    html_path = clip_dir / "clip.html"
    out_path = clip_dir / "clip.mp4"
    clip_id = clip.get("clip_id", "?")
    total_frames = clip.get("frames")

    last_errors = ["not called"]
    for attempt in range(1, MOTION_ATTEMPTS + 1):
        fragment = _strip_fence(stream_text(client, messages, model, label=f"motion_html {clip_id} attempt {attempt}"))
        errors = validate_motion_fragment(fragment)
        if not errors:
            html_path.write_text(wrap_motion_html(fragment, clip["duration_ms"], theme_css), encoding="utf-8")
            try:
                render_motion_html(html_path, out_path, clip["duration_ms"], total_frames)
                return out_path
            except StaticRenderError as e:
                errors = [str(e)]
        last_errors = errors
        print(f"  [{clip_id}] motion attempt {attempt}/{MOTION_ATTEMPTS} rejected: {errors}", file=sys.stderr)
        if fragment:
            messages.append({"role": "assistant", "content": fragment})
        messages.append(
            {
                "role": "user",
                "content": "上一次输出被拒绝。请修正下列问题后重新输出，只输出 #stage 内部内容：\n"
                + json.dumps({"errors": errors}, ensure_ascii=False, indent=2),
            }
        )
    raise RuntimeError(f"motion clip {clip_id} failed after {MOTION_ATTEMPTS} attempts: {last_errors}")


def read_text_file(path):
    path = Path(path)
    return path.read_text(encoding="utf-8") if path.exists() else ""


def load_art_style(code_dir):
    path = code_dir / "tools" / "prompts" / "art_style.json"
    if not path.exists():
        return {"image": "", "video": ""}
    cfg = load_json(path)
    return {"image": cfg.get("image", {}).get("styles", ""), "video": cfg.get("video", {}).get("style", "")}


def storyboard_text(clip):
    plan = clip.get("describe", {}).get("storyboard_plan", []) or []
    shots = [s.get("shot", "") for s in plan if isinstance(s, dict) and s.get("shot")]
    return "\n".join(f"- {s}" for s in shots)


def extract_last_frame_b64(video_path):
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-sseof", "-0.1", "-i", str(video_path),
                "-update", "1", "-q:v", "1", "-f", "image2pipe", "-vcodec", "png", "pipe:1",
            ],
            capture_output=True,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout:
            return base64.b64encode(result.stdout).decode()
    except Exception:  # noqa: BLE001
        pass
    return None


def _concat_videos(segment_paths, output_path):
    output_path = Path(output_path)
    if len(segment_paths) == 1:
        if Path(segment_paths[0]).resolve() != output_path.resolve():
            output_path.write_bytes(Path(segment_paths[0]).read_bytes())
        return output_path
    concat_list = output_path.parent / "concat_list.txt"
    concat_list.write_text("\n".join(f"file '{Path(p).resolve()}'" for p in segment_paths) + "\n", encoding="utf-8")
    result = run_cmd(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list), "-c", "copy", str(output_path)], timeout=120)
    if result.returncode != 0:
        result = run_cmd(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list), "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p", str(output_path)],
            timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed: {result.stderr[-500:]}")
    return output_path


def generate_seedance_clip(openrouter_key, clip, cast_entries, code_dir, clip_dir, force=False):
    clip_dir.mkdir(parents=True, exist_ok=True)
    cast_ids = clip.get("cast_ids", [])
    # All needed references for this clip: characters and environments/locations.
    ref_paths = cast_reference_images(code_dir, cast_entries, cast_ids, limit=None)
    art = load_art_style(code_dir)
    board = storyboard_text(clip)

    segments = clip.get("segments") or [
        {"start_ms": clip["start_ms"], "end_ms": clip["end_ms"], "narration": "", "continuation": False}
    ]
    one_segment = len(segments) == 1
    final = clip_dir / "clip.mp4"
    segment_paths = []
    prev_frame = None

    # Segment frame counts telescope inside the clip the same way clip frames telescope
    # inside the chapter, so the concatenated segments sum to exactly clip["frames"].
    clip_start_frame = clip.get("start_frame", round(clip["start_ms"] / 1000 * FPS))
    clip_end_frame = clip.get("end_frame", round(clip["end_ms"] / 1000 * FPS))
    prev_boundary = clip_start_frame

    for i, seg in enumerate(segments):
        seg_duration = (seg["end_ms"] - seg["start_ms"]) / 1000.0
        if seg_duration <= 0:
            continue
        is_last = i == len(segments) - 1
        seg_end_frame = clip_end_frame if is_last else round(seg["end_ms"] / 1000 * FPS)
        seg_frames = max(1, seg_end_frame - prev_boundary)
        prev_boundary = seg_end_frame
        seg_output = final if one_segment else clip_dir / f"seg_{i}.mp4"
        continuation = bool(seg.get("continuation"))

        # Resume: a segment already on disk at the right length is paid for. Reuse it
        # rather than re-billing a Seedance generation because a later segment failed.
        if not force and _clip_output_ok(seg_output, seg["end_ms"] - seg["start_ms"], seg_frames):
            print(f"    seg {i} already rendered ({seg_frames}f), reusing", file=sys.stderr)
            segment_paths.append(seg_output)
            prev_frame = extract_last_frame_b64(seg_output)
            continue

        # Each generation pass gets its own storyboard image, built from the
        # references needed for this segment.
        storyboard_path = clip_dir / ("storyboard.jpg" if one_segment else f"storyboard_{i}.jpg")
        try:
            input_refs = [{"type": "image_url", "image_url": {"url": image_to_data_url(p)}} for p in ref_paths]
            image_prompt = (
                f"{art['image']}\n"
                "为下面的 seedance 分镜画一张 16:9 单帧故事板参考图，无文字水印：\n"
                f"{board}\n"
                f"本段旁白：{seg.get('narration', '')}\n"
                + ("本段承接上一段末帧，保持同一主体、场景状态与镜头方向。\n" if continuation else "")
            )
            image_bytes = call_image_api(openrouter_key, image_prompt, input_references=input_refs or None, label=f"storyboard {clip['clip_id']} seg {i}")
            save_jpg(Image.open(io.BytesIO(image_bytes)), storyboard_path)
        except Exception as e:  # noqa: BLE001
            print(f"    storyboard {i} failed, continuing without it: {e}", file=sys.stderr)

        submit_refs = ([storyboard_path] if storyboard_path.exists() else []) + ref_paths
        prompt = (
            f"{art['video']}\n"
            f"clip 视觉大意：{clip.get('summary', '')}\n"
            f"分镜：\n{board}\n"
            f"近似运镜/节奏意图（推动生成，不保证逐帧）：\n{directives_text(clip)}\n"
            f"本段旁白：{seg.get('narration', '')}\n"
            + ("本段承接上一段末帧，同一主体、同一场景状态、同一镜头方向，不要重起场景。\n" if continuation else "")
            + "依据分镜/角色/环境参考图保持主体、构图与风格一致。成片静音，人物自然表演，不要正对镜头念稿。"
        )
        submit_seedance(
            openrouter_key,
            prompt,
            seg_duration,
            seg_output,
            reference_paths=submit_refs,
            first_frame_path=storyboard_path if (storyboard_path.exists() and not continuation) else None,
            prev_last_frame_b64=prev_frame if continuation else None,
            label=f"seedance {clip['clip_id']} seg {i}",
        )
        frame_snap(seg_output, seg_frames, seg_output)
        segment_paths.append(seg_output)
        prev_frame = extract_last_frame_b64(seg_output)

    if not segment_paths:
        raise RuntimeError("no seedance segments rendered")
    _concat_videos(segment_paths, final)
    return final


def process_generate(client, model, prompt, code_dir, project_dir, cast_entries, visual, manifest_path, force=False):
    """Generate every clip in a chapter, recording each outcome as it settles.

    A clip that cannot be generated is recorded as status "failed" instead of raising
    through the pool: one bad clip used to unwind process_generate and leave the chapter
    with no manifest at all, discarding every sibling clip that had already succeeded --
    including paid Seedance renders. The manifest is written before generation starts
    and rewritten as each clip finishes, so an interruption never loses the survivors.
    4_compose refuses any chapter whose manifest is not all-ok, which is what makes
    recording a failure safe rather than a way to ship a hole in the video.
    """
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
    chapter_dir = project_dir / "clips" / f"chapter_{visual['chapter']}"
    entries = {
        c["clip_id"]: {
            "clip_id": c["clip_id"],
            "method": c.get("method"),
            "video_file": str(chapter_dir / c["clip_id"] / "clip.mp4"),
            "duration_ms": c["duration_ms"],
            "frames": c.get("frames"),
            "status": "pending",
        }
        for c in visual["clips"]
    }
    lock = threading.Lock()

    def flush():
        with lock:
            write_json(manifest_path, {"chapter": visual["chapter"], "clips": [entries[c["clip_id"]] for c in visual["clips"]]})

    flush()

    def generate_one(clip):
        clip_id = clip["clip_id"]
        clip_dir = chapter_dir / clip_id
        if not force and _clip_output_ok(clip_dir / "clip.mp4", clip["duration_ms"], clip.get("frames")):
            print(f"  skip {clip_id} (already rendered)", file=sys.stderr)
            entries[clip_id].update(status="ok", reused=True)
            flush()
            return
        print(f"  generate {clip_id} ({clip['method']}, {clip['duration_ms']}ms)", file=sys.stderr)
        try:
            if clip["method"] == "seedance":
                if not openrouter_key:
                    raise RuntimeError("OPENROUTER_API_KEY not set")
                out = generate_seedance_clip(openrouter_key, clip, cast_entries, code_dir, clip_dir, force=force)
            else:
                out = generate_motion_clip(client, model, prompt, clip, clip_dir)
            entries[clip_id].update(status="ok", video_file=str(out))
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED {clip_id}: {e}", file=sys.stderr)
            entries[clip_id].update(status="failed", error=str(e))
        flush()

    # Clips are independent of each other (seedance segment chaining is internal
    # to each clip), so they generate in parallel; manifest stays in clip order.
    with ThreadPoolExecutor(max_workers=max(1, min(MAX_WORKERS, len(visual["clips"])))) as pool:
        list(pool.map(generate_one, visual["clips"]))
    return {"chapter": visual["chapter"], "clips": [entries[c["clip_id"]] for c in visual["clips"]]}


def _plan_without_internals(visual):
    """Plan as it should live on disk: runtime-only keys (_char_timings) stripped."""
    return {
        **{k: v for k, v in visual.items() if k != "clips"},
        "clips": [{k: v for k, v in c.items() if not k.startswith("_")} for c in visual["clips"]],
    }


def process_chapter(client, model, prompt, code_dir, project_dir, script_obj, chapter_id, voice, force=False):
    script_chapter = chapter_from_script(script_obj, chapter_id)
    chapter_name = script_chapter.get("title") or f"Chapter {chapter_id}"
    audio_dir = project_dir / "audio"
    plans_dir = project_dir / "clip_plans"
    plans_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}\nChapter {chapter_id}: {chapter_name}\n{'=' * 60}", file=sys.stderr)
    timed_lines = load_timed_lines(audio_dir, chapter_id, voice, script_chapter)
    chars_by_line = load_chars_by_line(audio_dir, chapter_id, voice, timed_lines)
    cast_set = load_cast_set(project_dir)
    cast_entries = available_cast(cast_set, chapter_id)
    cast_ids = {e.get("cast_id") for e in cast_entries}
    cast_text = format_cast(cast_entries)
    timed_script = format_timed_script(timed_lines, chars_by_line)

    route_path = plans_dir / f"route_chapter_{chapter_id}.json"
    if route_path.exists():
        route = load_json(route_path)
    else:
        messages = build_route_messages(prompt, timed_script, cast_text)
        route = call_llm_json(client, messages, model, validate_route_factory(timed_lines, cast_ids), f"route chapter {chapter_id}")
        route = {"chapter": chapter_id, "chapter_name": chapter_name, "voice": voice, "clips": route["clips"]}
    # Snap before describe so keyframe plans are validated against the frame-exact
    # duration the renderer will actually use. Idempotent, so cached routes get it too.
    snap_clips_to_frames(route["clips"])
    write_json(route_path, route)
    print(f"route: {len(route['clips'])} clips", file=sys.stderr)

    visual_path = plans_dir / f"visual_chapter_{chapter_id}.json"
    if visual_path.exists():
        visual = load_json(visual_path)
    else:
        def describe_one(clip):
            others = [
                {"clip_id": c.get("clip_id"), "method": c.get("method"), "summary": c.get("summary")}
                for c in route["clips"]
                if c.get("clip_id") != clip.get("clip_id")
            ]
            chapter_context = json.dumps(others, ensure_ascii=False, indent=2) if others else "（无）"
            messages = build_describe_messages(prompt, clip, chapter_context, cast_text)
            call_llm_json(
                client,
                messages,
                model,
                lambda parsed, clip=clip: validate_describe_for_clip(clip, parsed),
                f"describe {clip['clip_id']}",
            )

        # Per-clip describe calls are independent (coherence comes from the shared
        # route + chapter_context, not from seeing each other's answers).
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(route["clips"]))) as pool:
            list(pool.map(describe_one, route["clips"]))
        visual = {
            "chapter": chapter_id,
            "chapter_name": chapter_name,
            "voice": voice,
            "model": model,
            "method_counts": dict(Counter(c.get("method") for c in route["clips"])),
            "clips": route["clips"],
        }
        write_json(visual_path, _plan_without_internals(visual))
    print(f"describe: {len(visual['clips'])} clips", file=sys.stderr)

    # Snap again on the visual plan: when it is loaded from cache its clips come from
    # the file rather than the route object just snapped above, so a plan written before
    # frame telescoping existed would otherwise carry no frame counts and fall back to
    # per-clip rounding. Idempotent -- frames derive from end_ms, which never changes.
    snap_clips_to_frames(visual["clips"])
    attach_clip_char_timings(visual["clips"], chars_by_line)
    if any(c.get("method") == "seedance" and not c.get("segments") for c in visual["clips"]):
        split_seedance_clips(visual["clips"], client, model)
        # Persist the split so a re-run reuses the same cut points instead of asking
        # the LLM again and landing on different segment boundaries than the seg_N.mp4
        # files already on disk.
        write_json(visual_path, _plan_without_internals(visual))

    manifest_path = project_dir / "clips" / f"chapter_{chapter_id}" / "clips_manifest.json"
    manifest = process_generate(
        client, model, prompt, code_dir, project_dir, cast_entries, visual, manifest_path, force=force
    )
    print(f"Saved {manifest_path}", file=sys.stderr)
    failed = [c["clip_id"] for c in manifest["clips"] if c.get("status") != "ok"]
    if failed:
        raise RuntimeError(f"{len(failed)}/{len(manifest['clips'])} clips failed: {failed} (re-run to retry only these)")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Part 3: route -> describe -> generate clips")
    parser.add_argument("--project-id", required=True)
    # Not type=int: compared as a string against whatever script_final.json holds, so
    # this works on projects written before chapter_id was pinned to an integer.
    parser.add_argument("--chapter", default=None, help="chapter_id to build, exactly as written in script_final.json")
    parser.add_argument("--voice", choices=["female", "male"], default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--force", action="store_true", help="regenerate clips even if a valid clip.mp4 already exists")
    parser.add_argument("--prompt", default="tools/prompts/3_clips.json")
    args = parser.parse_args()

    code_dir = Path(__file__).parent.parent
    project_dir = code_dir / "projects" / args.project_id
    set_trace_dir(project_dir / "logs")
    load_dotenv(code_dir / ".env")
    voice = resolve_voice(code_dir, args.voice)

    script_path = project_dir / "script_final.json"
    if not script_path.exists():
        print(f"Error: script_final.json not found at {script_path}", file=sys.stderr)
        sys.exit(1)
    script_obj = load_json(script_path)

    prompt_path = code_dir / args.prompt
    if not prompt_path.exists():
        print(f"Error: prompt not found: {prompt_path}", file=sys.stderr)
        sys.exit(1)
    prompt = load_json(prompt_path)

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    api_base = os.environ.get("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not set in .env", file=sys.stderr)
        sys.exit(1)
    client = OpenAI(api_key=api_key, base_url=api_base, timeout=600)

    chapters = [ch.get("chapter_id", i) for i, ch in enumerate(script_obj.get("chapters", []) or [], 1) if isinstance(ch, dict)]
    if args.chapter is not None:
        selected = [c for c in chapters if str(c) == str(args.chapter)]
        if not selected:
            print(f"Error: chapter {args.chapter!r} not found; script has {chapters}", file=sys.stderr)
            sys.exit(1)
        chapters = selected

    failed = []
    for chapter_id in chapters:
        try:
            process_chapter(
                client, args.model, prompt, code_dir, project_dir, script_obj,
                chapter_id, voice, force=args.force,
            )
        except Exception as e:  # noqa: BLE001
            print(f"FAILED chapter {chapter_id}: {e}", file=sys.stderr)
            failed.append(chapter_id)
    if failed:
        print(f"\nFailed chapters: {failed}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
