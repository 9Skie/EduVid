#!/usr/bin/env python3
"""
Stage 6: Chapter to Audio (TTS via ElevenLabs v3).

Reads script_final.json, synthesizes one chapter at a time with
character-level timestamp alignment.

Usage (run from code/ with venv activated):
    source .venv/bin/activate
    python scripts/6_script_chapter_to_audio.py --chapter 1   # voice = teacher's narration_gender
    python scripts/6_script_chapter_to_audio.py               # all chapters
    python scripts/6_script_chapter_to_audio.py --voice male  # override the voice
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from elevenlabs import ElevenLabs


# --- Voice profiles ---

VOICES = {
    "female": {
        "voice_id": "APSIkVZudNbPAwyPoeVO",
        "name": "Female narrator",
    },
    "male": {
        "voice_id": "W8lBaQb9YIoddhxfQNLP",
        "name": "Male narrator",
    },
}

VOICE_SETTINGS = {
    "stability": 0.5,
    "similarity_boost": 0.75,
    "style": 0.3,
    "use_speaker_boost": True,
    "speed": 1.2,
}

MODEL_ID = "eleven_v3"
CLIENT_TIMEOUT = 600  # seconds — long chapters can take 5+ minutes


# --- Helpers ---

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_audio(client, text, voice_id):
    """Call ElevenLabs with timestamps. Returns (audio_bytes, alignment_dict)."""
    response = client.text_to_speech.convert_with_timestamps(
        voice_id=voice_id,
        text=text,
        model_id=MODEL_ID,
        voice_settings=VOICE_SETTINGS,
    )

    audio_bytes = base64.b64decode(response.audio_base_64)
    align = response.alignment

    if align is None:
        raise RuntimeError(
            f"ElevenLabs returned no alignment data for text ({len(text)} chars). "
            f"This is an API-level failure — the audio was generated but timing data is missing."
        )

    return audio_bytes, {
        "characters": list(align.characters),
        "char_start_times": list(align.character_start_times_seconds),
        "char_end_times": list(align.character_end_times_seconds),
    }


def compute_line_timings(alignment):
    """Split character-level alignment into sentence-level segments."""
    chars = alignment["characters"]
    starts = alignment["char_start_times"]
    ends = alignment["char_end_times"]

    if not chars:
        return []

    lines = []
    current_start = 0

    for i, ch in enumerate(chars):
        if ch in ("。", "！", "？", "\n") and i > current_start:
            line_text = "".join(chars[current_start:i + 1]).strip()
            if line_text:
                lines.append({
                    "start": round(starts[current_start], 3),
                    "end": round(ends[i], 3),
                    "text": line_text,
                })
            current_start = i + 1

    if current_start < len(chars):
        line_text = "".join(chars[current_start:]).strip()
        if line_text:
            lines.append({
                "start": round(starts[current_start], 3),
                "end": round(ends[-1], 3),
                "text": line_text,
            })

    return lines


def process_chapter(client, voice_cfg, voice_tag, chapter_num, chapter_name, text, audio_dir):
    """Synthesize one chapter. Returns result metadata dict."""
    print(f"\nChapter {chapter_num}: {chapter_name} ({len(text)} chars)", file=sys.stderr)

    if len(text) > 5000:
        print(f"Warning: {len(text)} chars exceeds v3 limit (5000). Truncating.", file=sys.stderr)
        text = text[:4990]

    t0 = time.time()
    audio_bytes, alignment = generate_audio(client, text, voice_cfg["voice_id"])
    elapsed = time.time() - t0

    tag = f"voice_chapter_{chapter_num}_{voice_tag}"
    audio_path = audio_dir / f"{tag}.mp3"
    align_path = audio_dir / f"{tag}_alignment.json"
    lines_path = audio_dir / f"{tag}_lines.json"

    audio_path.write_bytes(audio_bytes)
    save_json(alignment, align_path)
    lines = compute_line_timings(alignment)
    save_json(lines, lines_path)

    total = alignment["char_end_times"][-1] if alignment["char_end_times"] else 0
    print(f"  Audio: {audio_path.name} ({len(audio_bytes) // 1024} KB, {total:.1f}s)", file=sys.stderr)
    print(f"  Lines: {len(lines)} segments", file=sys.stderr)
    print(f"  API time: {elapsed:.1f}s", file=sys.stderr)

    return {
        "chapter": chapter_num,
        "name": chapter_name,
        "voice": voice_cfg["name"],
        "audio": str(audio_path.name),
        "alignment": str(align_path.name),
        "lines": str(lines_path.name),
        "duration_s": round(total, 2),
        "char_count": len(text),
    }


def write_chapter_manifest(audio_dir, chapter_num, voice_tag, voice_cfg, result):
    """Write this chapter's own manifest (manifest_chapter_N.json).

    Each chapter owns a separate file, so chapters can be synthesized
    concurrently without racing on a shared index. The shape mirrors the old
    aggregate manifest (voice-keyed, with a `chapters` list) but holds exactly
    this one chapter, so readers can merge the per-chapter files trivially.
    """
    key = f"voice_{voice_tag}"
    manifest = {
        key: {
            "voice_id": voice_cfg["voice_id"],
            "voice_name": voice_cfg["name"],
            "model": MODEL_ID,
            "settings": VOICE_SETTINGS,
            "chapters": [result],
            "total_duration_s": result["duration_s"],
        }
    }
    manifest_path = audio_dir / f"manifest_chapter_{chapter_num}.json"
    save_json(manifest, manifest_path)
    return manifest_path


# --- Main ---

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
    parser = argparse.ArgumentParser(description="Generate TTS audio with timestamps per chapter")
    parser.add_argument("--script", default="tmp/script_final.json",
                        help="Path to final script JSON")
    parser.add_argument("--chapter", type=int, default=None,
                        help="Process a single chapter by number (1-indexed). If omitted, processes all.")
    parser.add_argument("--voice", choices=["female", "male"], default=None,
                        help="Override narrator voice (default: teacher's narration_gender)")
    args = parser.parse_args()

    project_dir = Path(__file__).parent.parent
    load_dotenv(project_dir / ".env")
    args.voice = resolve_voice(project_dir, args.voice)

    api_key = os.environ.get("ELEVENLABS_API_KEY", "")
    if not api_key:
        print("Error: ELEVENLABS_API_KEY not set in .env", file=sys.stderr)
        sys.exit(1)

    voice_cfg = VOICES[args.voice]
    print(f"Voice: {voice_cfg['name']} ({voice_cfg['voice_id']})", file=sys.stderr)
    print(f"Model: {MODEL_ID}", file=sys.stderr)
    print(f"Settings: {VOICE_SETTINGS}", file=sys.stderr)

    script_path = project_dir / args.script
    if not script_path.exists():
        print(f"Error: {script_path} not found. Run stage 4 first.", file=sys.stderr)
        sys.exit(1)

    script_data = load_json(script_path)
    chapters = script_data.get("chapters", [])
    if not chapters:
        print("Error: no chapters found in script JSON", file=sys.stderr)
        sys.exit(1)

    client = ElevenLabs(api_key=api_key, timeout=CLIENT_TIMEOUT)
    audio_dir = project_dir / "tmp" / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Pick which chapters to process
    if args.chapter is not None:
        if args.chapter < 1 or args.chapter > len(chapters):
            print(f"Error: chapter {args.chapter} out of range (1-{len(chapters)})", file=sys.stderr)
            sys.exit(1)
        targets = [(args.chapter, chapters[args.chapter - 1])]
    else:
        targets = [(i + 1, ch) for i, ch in enumerate(chapters)]

    written = []
    for num, ch in targets:
        try:
            result = process_chapter(
                client, voice_cfg, args.voice, num,
                ch.get("name", "?"), ch.get("script", ""),
                audio_dir,
            )
            written.append(
                write_chapter_manifest(audio_dir, num, args.voice, voice_cfg, result)
            )
        except Exception as e:
            print(f"\nFAILED on chapter {num}: {e}", file=sys.stderr)
            print("Previously finished chapters keep their own manifests.", file=sys.stderr)
            sys.exit(1)

    print(f"\n{'='*50}", file=sys.stderr)
    print(f"Done. {len(targets)} chapter(s) synthesized.", file=sys.stderr)
    for p in written:
        print(f"Manifest: {p}", file=sys.stderr)


if __name__ == "__main__":
    main()
