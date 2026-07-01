#!/usr/bin/env python3
"""
Step 10: Assemble final video.

Per chapter: concat the chapter's clips, slap the chapter audio on top.
Full: concat chapter_1..chapter_N into one big video.

Usage (run from code/ with venv activated):
    python scripts/10_clip_to_video.py --chapter 1
    python scripts/10_clip_to_video.py --all
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
TMP_DIR = PROJECT_DIR / "tmp"
OUT_DIR = PROJECT_DIR / "out"
# String templates (filled per chapter via .format(n=...)); wrap in Path() at
# the call site. They're not Path objects — Path has no .format().
CLIPS_DIR = str(TMP_DIR / "clips" / "chapter_{n}")
VISUAL_FLOW_PATH = str(TMP_DIR / "visual_flow" / "visual_flow_chapter_{n}.json")
AUDIO_DIR = TMP_DIR / "audio"

# Target spec for assembly. Seedance is locked at ~24fps, so the pipeline
# renders everything at 24 (see stage 9); normalize to 24 here too so no clip
# gets fps-resampled. Normalization still runs to fix timebase / SAR / embedded
# audio differences and any stray old 30fps clips.
FPS = 24
OUT_W = 1280
OUT_H = 720


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_name(s):
    return re.sub(r"[^a-zA-Z0-9_]", "_", str(s))


def ffprobe_duration(path):
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(r.stdout.strip())
    except (ValueError, AttributeError):
        return None


def find_audio(chapter):
    manifest_path = AUDIO_DIR / f"manifest_chapter_{chapter}.json"
    if not manifest_path.exists():
        return None
    manifest = load_json(manifest_path)
    for voice_data in manifest.values():
        for ch in voice_data.get("chapters", []):
            if ch.get("chapter") == chapter:
                return AUDIO_DIR / ch["audio"]
    return None


def write_concat_list(paths, list_path):
    lines = [f"file '{p.resolve()}'" for p in paths]
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_ffmpeg(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr[-1500:]}")


def assemble_chapter(n):
    visual_flow_path = Path(VISUAL_FLOW_PATH.format(n=n))
    if not visual_flow_path.exists():
        print(f"Error: {visual_flow_path} not found", file=sys.stderr)
        return False
    vf = load_json(visual_flow_path)

    clips = sorted(vf["clips"], key=lambda c: c.get("start_ms", 0))
    clips_dir = Path(CLIPS_DIR.format(n=n))
    clip_paths = [
        clips_dir / safe_name(c["clip_id"]) / "clip.mp4"
        for c in clips
    ]
    missing = [p for p in clip_paths if not p.exists()]
    if missing:
        print(f"Error: {len(missing)} clip(s) missing for chapter {n}:", file=sys.stderr)
        for p in missing:
            print(f"  {p}", file=sys.stderr)
        return False

    audio = find_audio(n)
    if not audio:
        print(f"Error: no audio for chapter {n} (missing manifest_chapter_{n}.json)", file=sys.stderr)
        return False

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUT_DIR / f"chapter_{n}.mp4"

    # Use the concat FILTER (not the demuxer): normalize each clip to the same
    # fps / size / SAR and take video-only ([i:v], dropping clips' own audio),
    # then mux the chapter narration. Robust to heterogeneous inputs.
    cmd = ["ffmpeg", "-y"]
    for p in clip_paths:
        cmd += ["-i", str(p)]
    cmd += ["-i", str(audio)]

    norm = (
        f"fps={FPS},scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease,"
        f"pad={OUT_W}:{OUT_H}:(ow-iw)/2:(oh-ih)/2,setsar=1"
    )
    nclips = len(clip_paths)
    filt = "".join(f"[{i}:v]{norm}[v{i}];" for i in range(nclips))
    filt += "".join(f"[v{i}]" for i in range(nclips))
    filt += f"concat=n={nclips}:v=1:a=0[outv]"

    print(f"Chapter {n} → {output}", file=sys.stderr)
    try:
        run_ffmpeg([
            *cmd,
            "-filter_complex", filt,
            "-map", "[outv]",
            "-map", f"{nclips}:a",
            "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            str(output),
        ])
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return False

    dur = ffprobe_duration(output)
    print(f"  ✓ chapter_{n}.mp4 ({dur:.2f}s)", file=sys.stderr)
    return True


def assemble_full():
    chapter_videos = sorted(
        OUT_DIR.glob("chapter_*.mp4"),
        key=lambda p: int(re.search(r"chapter_(\d+)", p.name).group(1)),
    )
    if not chapter_videos:
        print(f"Error: no chapter videos in {OUT_DIR}", file=sys.stderr)
        return False

    list_path = OUT_DIR / ".concat_full.txt"
    write_concat_list(chapter_videos, list_path)
    output = OUT_DIR / "full.mp4"

    print(f"\nFull video → {output}", file=sys.stderr)
    try:
        run_ffmpeg([
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", str(list_path),
            "-c", "copy",
            str(output),
        ])
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return False
    finally:
        list_path.unlink(missing_ok=True)

    dur = ffprobe_duration(output)
    print(f"  ✓ full.mp4 ({dur:.2f}s, {len(chapter_videos)} chapters)", file=sys.stderr)
    return True


def main():
    ap = argparse.ArgumentParser(description="Assemble chapter or full video")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--chapter", type=int)
    g.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if args.chapter:
        ok = assemble_chapter(args.chapter)
        sys.exit(0 if ok else 1)

    # --all: assemble every chapter that has a visual flow, then concat
    chapter_ns = sorted(
        int(re.search(r"(\d+)", p.name).group(1))
        for p in (TMP_DIR / "visual_flow").glob("visual_flow_chapter_*.json")
    )
    failed = [n for n in chapter_ns if not assemble_chapter(n)]
    if failed:
        print(f"\n{len(failed)} chapter(s) failed: {failed}", file=sys.stderr)
        print("Skipping full assembly.", file=sys.stderr)
        sys.exit(1)
    ok = assemble_full()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
