#!/usr/bin/env python3
"""
Step 4: compose clips + narration into chapter videos and a full video.

Inputs under projects/<project_id>/:
  clips/chapter_N/clips_manifest.json     (from 3_clips.py; all statuses must be ok)
  audio/voice_chapter_N_{voice}.mp3
  audio/voice_chapter_N_{voice}_lines.json
  script_final.json

Outputs under projects/<project_id>/out/:
  chapter_N.mp4
  full.mp4              (when composing all chapters)
  full.vtt, full.srt    (when composing all chapters)

Usage (run from code/ with venv activated):
    source .venv/bin/activate
    python scripts/4_compose.py --project-id triangles
    python scripts/4_compose.py --project-id triangles --chapter 1
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv


FPS = 24
WIDTH = 1280
HEIGHT = 720


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")


def run_cmd(cmd, timeout=600):
    return subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=timeout, check=False)


def ffprobe(path):
    proc = run_cmd(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate:format=duration",
            "-of", "json", str(path),
        ],
        timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {proc.stderr[-300:]}")
    data = json.loads(proc.stdout or "{}")
    stream = (data.get("streams") or [{}])[0]
    fps = stream.get("r_frame_rate", "0/1")
    try:
        num, den = fps.split("/")
        fps_value = float(num) / float(den)
    except Exception:  # noqa: BLE001
        fps_value = 0.0
    return {
        "width": stream.get("width"),
        "height": stream.get("height"),
        "fps": fps_value,
        "duration": float(data.get("format", {}).get("duration", 0.0) or 0.0),
    }


def resolve_voice(code_dir, cli_voice):
    if cli_voice:
        return cli_voice
    try:
        pre = load_json(code_dir / "tools" / "prompts" / "1_preproduction.json")
        gender = pre.get("teacher_job", {}).get("narration_gender", "female")
        return gender if gender in ("male", "female") else "female"
    except Exception:  # noqa: BLE001
        return "female"


def read_chapter_inputs(project_dir, chapter_id, voice):
    manifest_path = project_dir / "clips" / f"chapter_{chapter_id}" / "clips_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"clips manifest not found: {manifest_path}; run 3_clips.py first")
    manifest = load_json(manifest_path)
    clips = manifest.get("clips", []) or []
    if not clips:
        raise RuntimeError(f"no clips in {manifest_path}")

    video_paths = []
    not_ok = [c for c in clips if c.get("status") != "ok"]
    if not_ok:
        detail = ", ".join(f"{c.get('clip_id')}={c.get('status')}" for c in not_ok)
        raise RuntimeError(
            f"chapter {chapter_id}: {len(not_ok)}/{len(clips)} clips are not ok ({detail}). "
            f"Re-run 3_clips.py --chapter {chapter_id} to retry just those; composing now "
            f"would silently drop them from the video."
        )
    for item in clips:
        path = Path(item.get("video_file", ""))
        if not path.is_absolute():
            path = project_dir / path
        if not path.exists():
            raise FileNotFoundError(f"clip video not found: {path}")
        meta = ffprobe(path)
        if meta["width"] != WIDTH or meta["height"] != HEIGHT or abs(meta["fps"] - FPS) > 0.01:
            raise RuntimeError(
                f"clip {item.get('clip_id')} is {meta['width']}x{meta['height']}@{meta['fps']:.3f}, expected {WIDTH}x{HEIGHT}@{FPS}"
            )
        video_paths.append(path)

    audio_path = project_dir / "audio" / f"voice_chapter_{chapter_id}_{voice}.mp3"
    lines_path = project_dir / "audio" / f"voice_chapter_{chapter_id}_{voice}_lines.json"
    if not audio_path.exists():
        raise FileNotFoundError(f"chapter audio not found: {audio_path}")
    if not lines_path.exists():
        raise FileNotFoundError(f"chapter timed lines not found: {lines_path}")
    lines = load_json(lines_path)
    return video_paths, audio_path, lines


def concat_videos(inputs, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    list_path = output.parent / f"{output.stem}_concat.txt"
    list_path.write_text("".join(f"file '{p.resolve()}'\n" for p in inputs), encoding="utf-8")
    proc = run_cmd(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path), "-c", "copy", str(output)])
    if proc.returncode == 0:
        return
    proc = run_cmd(
        [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path),
            "-vf", f"scale={WIDTH}:{HEIGHT}", "-r", str(FPS),
            "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p", str(output),
        ],
        timeout=900,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg concat failed for {output}: {proc.stderr[-500:]}")


def mux_audio(video_path, audio_path, output):
    proc = run_cmd(
        [
            "ffmpeg", "-y", "-i", str(video_path), "-i", str(audio_path),
            "-c:v", "copy", "-c:a", "aac", "-shortest", str(output),
        ],
        timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg mux failed for {output}: {proc.stderr[-500:]}")


def ms_to_clock(ms, srt=False):
    ms = max(0, int(round(ms)))
    h = ms // 3600000
    ms %= 3600000
    m = ms // 60000
    ms %= 60000
    s = ms // 1000
    ms %= 1000
    sep = "," if srt else "."
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def chapter_duration_ms(lines):
    ends = [line.get("end_ms") for line in lines if isinstance(line, dict) and line.get("end_ms") is not None]
    return max(ends) if ends else 0


def compose_chapter(project_dir, chapter_id, voice, out_dir):
    videos, audio, lines = read_chapter_inputs(project_dir, chapter_id, voice)
    silent = out_dir / f"chapter_{chapter_id}_silent.mp4"
    final = out_dir / f"chapter_{chapter_id}.mp4"
    concat_videos(videos, silent)
    mux_audio(silent, audio, final)
    meta = ffprobe(final)
    audio_ms = chapter_duration_ms(lines)
    video_ms = round(meta["duration"] * 1000)
    # With frame telescoping in 3_clips.py these should agree to well under a frame.
    # A larger gap means the clips no longer match the plan that timed them.
    if audio_ms and abs(video_ms - audio_ms) > 500:
        print(
            f"Warning: chapter {chapter_id} video is {video_ms}ms but narration is {audio_ms}ms "
            f"({video_ms - audio_ms:+d}ms)",
            file=sys.stderr,
        )
    return {
        "chapter": chapter_id,
        "video": final,
        "duration_s": meta["duration"],
        "duration_ms": video_ms,
        "lines": lines,
        "audio_ms": audio_ms,
    }


def write_captions(chapter_results, out_dir):
    vtt = ["WEBVTT", ""]
    srt = []
    offset = 0
    idx = 1
    for result in chapter_results:
        for line in result["lines"]:
            if not isinstance(line, dict):
                continue
            start = line.get("start_ms")
            end = line.get("end_ms")
            text = (line.get("text") or "").strip()
            if start is None or end is None or not text:
                continue
            s = offset + start
            e = offset + end
            vtt.append(f"{ms_to_clock(s)} --> {ms_to_clock(e)}")
            vtt.append(text)
            vtt.append("")
            srt.append(str(idx))
            srt.append(f"{ms_to_clock(s, srt=True)} --> {ms_to_clock(e, srt=True)}")
            srt.append(text)
            srt.append("")
            idx += 1
        # Advance by the chapter video's MEASURED length, not by where its narration
        # ended. full.mp4 is a concatenation, so chapter N+1 starts at the sum of the
        # real durations; offsetting by narration end lets captions drift chapter over
        # chapter against the picture they belong to.
        offset += result.get("duration_ms") or result["audio_ms"]
    write_text(out_dir / "full.vtt", "\n".join(vtt).rstrip() + "\n")
    write_text(out_dir / "full.srt", "\n".join(srt).rstrip() + "\n")


def main():
    parser = argparse.ArgumentParser(description="Step 10: compose final videos")
    parser.add_argument("--project-id", required=True)
    # Not type=int: compared as a string against whatever script_final.json holds, so
    # this works on projects written before chapter_id was pinned to an integer.
    parser.add_argument("--chapter", default=None, help="chapter_id to compose, exactly as written in script_final.json")
    parser.add_argument("--voice", choices=["female", "male"], default=None)
    args = parser.parse_args()

    code_dir = Path(__file__).parent.parent
    project_dir = code_dir / "projects" / args.project_id
    out_dir = project_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    load_dotenv(code_dir / ".env")
    voice = resolve_voice(code_dir, args.voice)

    script_obj = load_json(project_dir / "script_final.json")
    chapters = [ch.get("chapter_id", i) for i, ch in enumerate(script_obj.get("chapters", []) or [], 1) if isinstance(ch, dict)]
    if args.chapter is not None:
        selected = [c for c in chapters if str(c) == str(args.chapter)]
        if not selected:
            print(f"Error: chapter {args.chapter!r} not found; script has {chapters}", file=sys.stderr)
            sys.exit(1)
        chapters = selected

    results = []
    try:
        for chapter_id in chapters:
            print(f"compose chapter {chapter_id}...", file=sys.stderr)
            results.append(compose_chapter(project_dir, chapter_id, voice, out_dir))

        if args.chapter is None and len(results) > 1:
            full = out_dir / "full.mp4"
            concat_videos([r["video"] for r in results], full)
            write_captions(results, out_dir)
            full_meta = ffprobe(full)
            expected_s = sum(r["audio_ms"] for r in results) / 1000.0
            if abs(full_meta["duration"] - expected_s) > 0.5:
                print(
                    f"Warning: full duration {full_meta['duration']:.2f}s differs from narration {expected_s:.2f}s",
                    file=sys.stderr,
                )
            print(f"Saved {full}", file=sys.stderr)
            print(f"Saved {out_dir / 'full.vtt'}", file=sys.stderr)
            print(f"Saved {out_dir / 'full.srt'}", file=sys.stderr)
        elif args.chapter is None and len(results) == 1:
            write_captions(results, out_dir)
            print(f"Saved {out_dir / 'full.vtt'}", file=sys.stderr)
            print(f"Saved {out_dir / 'full.srt'}", file=sys.stderr)
    except Exception as e:  # noqa: BLE001
        print(f"compose failed: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
