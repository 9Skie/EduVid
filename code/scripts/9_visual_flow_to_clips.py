#!/usr/bin/env python3
"""
Step 9: Visual flow → clips

Takes per-clip visual designs from step 8 and renders each into a video file.

Three paths:
  - seedance  → OpenRouter video API (bytedance/seedance-2.0)
  - manim     → Python code-gen loop (generate → render → fix)
  - remotion  → TSX code-gen loop (generate → render → fix)

The code-gen loop: skill pick → inject skills + generate → render → check
duration → fix via search/replace edits. Up to 3 rounds. On failure, trim
the best attempt to exact duration (or leave gap if too short; black video
only if no attempt ever rendered).

Usage (run from code/ with venv activated):
    python scripts/9_visual_flow_to_clips.py --chapter 1
    python scripts/9_visual_flow_to_clips.py --chapter 1 --clip-ids 3,7
"""

import argparse
import base64
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml
import requests
from dotenv import load_dotenv
from openai import OpenAI

# ── Paths ─────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_DIR / "scripts"
REF_DIR = PROJECT_DIR.parent / "ref"
PROMPTS_DIR = PROJECT_DIR / "prompts"
CASTS_DIR = PROJECT_DIR / "casts"
TMP_DIR = PROJECT_DIR / "tmp"
# The remotion build project (with node_modules) lives under code/, not tmp/,
# so clearing tmp/ between runs doesn't wipe the one-time npm install.
REMOTION_PROJECT_DIR = PROJECT_DIR / "remotion_project"

MANIM_SKILLS_DIR = REF_DIR / "manim_skill" / "skills"
REMOTION_SKILLS_DIR = REF_DIR / "remotion_skill" / "skills"
THEME_PY = SCRIPTS_DIR / "theme.py"
THEME_TS = SCRIPTS_DIR / "theme.ts"

TEX_BIN = "/Library/TeX/texbin"

# ── Constants ─────────────────────────────────────────────────────────
MAX_ROUNDS = 3
RENDER_TIMEOUT = 180
# Seedance outputs ~24fps and we can't change that, so render manim/remotion/
# black at 24 too — uniform fps means no resampling (judder) at assembly.
FPS = 24
RESOLUTION_W = 1280
RESOLUTION_H = 720
# A code clip is accepted if its render lands within this of the target, then
# frame-snapped to EXACTLY round(dur*FPS) frames. Wider than half a frame so
# manim's per-animation frame rounding (a frame or two off) passes on round 0
# instead of burning the fix loop; off by more than this is a real timing bug.
SNAP_WINDOW_S = 0.5

# ── I/O helpers ───────────────────────────────────────────────────────


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ── Skill management ──────────────────────────────────────────────────


# Each backend has ONE skill; its docs are the real unit of choice. Per clip we
# load: SKILL.md overview + an always-on CORE_RULES subset + dynamically-picked
# docs. manim/remotion keep docs under rules/; seedance under references/. The
# docs have no frontmatter, so we read the first '# heading' as the title.

SKILL_ROOT = {
    "manim": MANIM_SKILLS_DIR / "manimce-best-practices",
    "remotion": REMOTION_SKILLS_DIR / "remotion",
    "seedance": REF_DIR / "seedance_skill",
}

DOC_SUBDIR = {"manim": "rules", "remotion": "rules", "seedance": "references"}

CORE_RULES = {
    "manim": ["scenes", "mobjects", "animations", "timing", "text", "positioning"],
    "remotion": ["timing", "sequencing", "text-animations", "video-layout", "api-reference"],
    # Seedance is prompt-craft, not code: always load the directing core.
    "seedance": [
        "directing-engine",
        "cinematography-shot-language",
        "anti-slop-lexicon",
        "prompt-compiler",
    ],
}

# Forced in when a clip chains (multi-segment or continuation frame), so the
# segments read as one continuous take.
SEEDANCE_CONTINUITY_DOCS = ["first-last-frame-guide", "continuation-handoff"]


def _doc_title_desc(path):
    """First markdown heading → title; first paragraph after it → description."""
    title = path.stem
    desc = []
    seen_heading = False
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            if desc:
                break
            continue
        if s.startswith("#"):
            if not seen_heading:
                title = s.lstrip("# ").strip()
                seen_heading = True
            continue
        desc.append(s)
        if len(" ".join(desc)) > 200:
            break
    return title, " ".join(desc)[:240]


def core_content(backend):
    """SKILL.md overview + always-on core docs — loaded for every call."""
    root = SKILL_ROOT[backend]
    sub = DOC_SUBDIR[backend]
    parts = []
    skill_md = root / "SKILL.md"
    if skill_md.exists():
        parts.append(
            f"### {backend} best-practices overview\n{skill_md.read_text(encoding='utf-8')}"
        )
    for name in CORE_RULES.get(backend, []):
        p = root / sub / f"{name}.md"
        if p.exists():
            parts.append(f"### {name}\n{p.read_text(encoding='utf-8')}")
    return "\n\n".join(parts)


def build_doc_index(backend):
    """Index the NON-core docs (the dynamic pool) by title + description."""
    root = SKILL_ROOT[backend]
    skip = set(CORE_RULES.get(backend, []))
    if backend == "seedance":
        # continuity docs are force-loaded on chaining, not pickable.
        skip |= set(SEEDANCE_CONTINUITY_DOCS)
    index = []
    docs_dir = root / DOC_SUBDIR[backend]
    if docs_dir.exists():
        for p in sorted(docs_dir.glob("*.md")):
            if p.stem in skip:
                continue
            title, desc = _doc_title_desc(p)
            index.append({"name": p.stem, "title": title, "description": desc})
    return index


def format_doc_index(docs):
    if not docs:
        return "(none)"
    lines = ["Additional rules docs (pick those relevant to this clip):"]
    for d in docs:
        lines.append(f"- {d['name']}: {d['title']} — {d['description']}")
    return "\n".join(lines)


def load_picked_content(backend, picked_rules):
    root = SKILL_ROOT[backend]
    sub = DOC_SUBDIR[backend]
    parts = []
    for name in picked_rules:
        p = root / sub / f"{name}.md"
        if p.exists():
            parts.append(f"### {name}\n{p.read_text(encoding='utf-8')}")
    return "\n\n".join(parts) if parts else ""


# ── Cast images ───────────────────────────────────────────────────────


def find_cast_images(cast_id):
    """Design-quality reference images for a cast member.

    Reads the cast JSON's `reference_images` (canonical) and prefers
    *_design.* over *_example.*. Falls back to example only when no
    design sheet exists (e.g. environment locations like alexandria).
    """
    cast_json = next(CASTS_DIR.rglob(f"{cast_id}.json"), None)
    if not cast_json:
        return []
    try:
        cast_data = json.loads(cast_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    refs = cast_data.get("reference_images", [])

    def existing(paths):
        return [PROJECT_DIR / p for p in paths if (PROJECT_DIR / p).exists()]

    design = existing(p for p in refs if "_design" in p)
    if design:
        return design
    non_example = existing(p for p in refs if "_example" not in p)
    if non_example:
        return non_example
    return existing(refs)


def images_to_b64(image_paths):
    refs = []
    for img in image_paths[:3]:
        b64 = base64.b64encode(img.read_bytes()).decode()
        ext = img.suffix.lstrip(".")
        refs.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/{ext};base64,{b64}"},
            }
        )
    return refs


# ── Time conversion ───────────────────────────────────────────────────


def convert_to_clip_relative(clip):
    ct = clip.get("char_timings", [])
    if not ct:
        return clip
    clip_start = ct[0]["start"]
    rel_beats = []
    for beat in clip.get("visual_design", {}).get("beats", []):
        rb = dict(beat)
        rb["t"] = [beat["t"][0] - clip_start, beat["t"][1] - clip_start]
        rel_beats.append(rb)
    rel_ct = []
    for c in ct:
        rc = dict(c)
        rc["start"] = round(c["start"] - clip_start, 3)
        rc["end"] = round(c["end"] - clip_start, 3)
        rel_ct.append(rc)
    clip["_rel_beats"] = rel_beats
    clip["_rel_char_timings"] = rel_ct
    clip["_clip_start"] = clip_start
    return clip


def clip_target_duration(clip):
    """Clip length in seconds from its real timing — no magic default."""
    if clip.get("duration_ms"):
        return clip["duration_ms"] / 1000.0
    if "start_ms" in clip and "end_ms" in clip:
        return (clip["end_ms"] - clip["start_ms"]) / 1000.0
    ct = clip.get("char_timings", [])
    if ct:
        return round(ct[-1]["end"] - ct[0]["start"], 3)
    return 0.0


# ── Formatting ────────────────────────────────────────────────────────


def format_beats(beats):
    if not beats:
        return "(no beats)"
    lines = []
    for i, b in enumerate(beats):
        t = b.get("t", [0, 0])
        lines.append(f"  [{i}] t=[{t[0]:.2f}, {t[1]:.2f}]s: {b.get('action', '?')}")
    return "\n".join(lines)


def format_char_timings(ct):
    if not ct:
        return "(no char timings)"
    chars = " ".join(f'{c["char"]}({c["start"]:.2f})' for c in ct)
    return chars


def build_chapter_context(clips, current_clip_id):
    lines = []
    for clip in clips:
        cid = clip.get("clip_id", "?")
        method = clip.get("method", "?")
        desc = clip.get("description", "")
        design = clip.get("visual_design", {})
        if design.get("kind") == "beats":
            beat_summary = "; ".join(
                b.get("action", "")[:40] for b in design.get("beats", [])
            )
            detail = f"beats: {beat_summary}"
        elif design.get("kind") == "shot":
            detail = f"shot: {design.get('prompt', '')[:60]}"
        else:
            detail = ""
        marker = "  <<< CURRENT" if cid == current_clip_id else ""
        lines.append(f"  Clip {cid} ({method}): {desc[:60]} — {detail}{marker}")
    return "\n".join(lines)


# ── LLM call ──────────────────────────────────────────────────────────


def make_client(cfg):
    api_key = os.environ.get(cfg["api_key_env"], "")
    if not api_key:
        print(f"Error: {cfg['api_key_env']} not set in .env", file=sys.stderr)
        sys.exit(1)
    api_base = os.environ.get(cfg["api_base_env"], cfg["api_base_default"])
    return OpenAI(api_key=api_key, base_url=api_base, timeout=600)


def llm_call(client, messages, cfg, json_mode=False):
    kwargs = dict(
        model=cfg["model"],
        messages=messages,
        max_tokens=cfg["max_tokens"],
        stream=True,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    stream = client.chat.completions.create(**kwargs)
    parts = []
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            print(reasoning, end="", flush=True, file=sys.stderr)
        if delta.content:
            parts.append(delta.content)
            print(delta.content, end="", flush=True, file=sys.stderr)
    print(file=sys.stderr)
    return "".join(parts)


# ── Code extraction ───────────────────────────────────────────────────


def extract_code(text, language):
    pattern = rf"```{language}\n?(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    if not text.strip().startswith("<"):
        lines = [l for l in text.strip().split("\n") if not l.startswith("```")]
        if lines:
            return "\n".join(lines)
    return None


def extract_code_with_retries(client, cfg, messages, text, language, max_retries=2):
    code = extract_code(text, language)
    if code:
        return code
    for attempt in range(max_retries):
        print(f"  [extract retry {attempt + 1}]", file=sys.stderr)
        retry_msg = messages + [
            {
                "role": "assistant",
                "content": text,
            },
            {
                "role": "user",
                "content": f"Please output ONLY the code wrapped in ```{language} fences. Use the exact same code from your previous response — no content editing, no explanation.",
            },
        ]
        resp = llm_call(client, retry_msg, cfg)
        code = extract_code(resp, language)
        if code:
            return code
    return None


# ── Edit parsing + application ────────────────────────────────────────


def parse_edits(text):
    edits = []
    pattern = r"<edit>\s*SEARCH:\s*(.*?)\s*REPLACE:\s*(.*?)\s*</edit>"
    for m in re.finditer(pattern, text, re.DOTALL | re.IGNORECASE):
        edits.append((m.group(1), m.group(2)))
    return edits


def apply_edits(code, edits):
    applied = 0
    for search, replace in edits:
        if search.strip() and search in code:
            code = code.replace(search, replace, 1)
            applied += 1
        else:
            print(f"  [edit skipped: SEARCH not found]", file=sys.stderr)
    return code, applied


def sanitize_manim_code(code):
    """Strip any LLM-added manim config override. The pipeline fixes the fps via
    the render command's --fps; a stray `config.frame_rate=` in the script fights
    it and intermittently yields 25fps. Drops every `config.<attr> =` assignment
    (resolution/fps are the pipeline's job); a now-unused config import is inert."""
    kept = [ln for ln in code.splitlines()
            if not re.match(r"\s*config\.\w+\s*=", ln)]
    return "\n".join(kept)


# ── Skill picking ─────────────────────────────────────────────────────


def llm_pick_docs(client, prompt_cfg, kimi_cfg, clip, doc_index, picked_so_far, error):
    task_template = prompt_cfg["skill_pick"]["task"]
    beats = clip.get("_rel_beats", clip.get("visual_design", {}).get("beats", []))
    beats_summary = "; ".join(b.get("action", "")[:50] for b in beats[:5])

    prev_section = ""
    if picked_so_far:
        prev_section = f"\n## Already loaded\n{', '.join(picked_so_far)}\n"

    error_section = ""
    if error:
        error_section = f"\n## Error from last attempt (you may need different docs)\n{error[:500]}\n"

    task = task_template.format(
        backend=clip["method"],
        skill_index=format_doc_index(doc_index),
        implementation=clip.get("description", ""),
        beats_summary=beats_summary,
        previous_picks_section=prev_section,
        error_section=error_section,
    )

    messages = [
        {"role": "system", "content": prompt_cfg["skill_pick"]["system"]},
        {"role": "user", "content": task},
    ]

    raw = llm_call(client, messages, kimi_cfg, json_mode=True)
    try:
        result = json.loads(raw)
        return result.get("docs", result.get("skills", [])) or []
    except json.JSONDecodeError:
        print("  [doc pick parse failed; core only]", file=sys.stderr)
        return []


# ── Rendering: manim ──────────────────────────────────────────────────


def render_manim(scene_file, scene_class, workspace):
    media_dir = workspace / "media"
    # The scene does `from theme import *`; manim runs with cwd=workspace, so the
    # frozen theme must sit there or the import fails and the clip falls back to
    # bare manim defaults (white bg, no design system).
    (workspace / "theme.py").write_text(
        THEME_PY.read_text(encoding="utf-8"), encoding="utf-8"
    )
    env = os.environ.copy()
    env["PATH"] = TEX_BIN + ":" + env.get("PATH", "")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "manim",
            "-qm",
            "--fps",
            str(FPS),  # -qm defaults to 30fps; force 24 to match seedance
            "--media_dir",
            str(media_dir),
            str(scene_file),
            scene_class,
        ],
        capture_output=True,
        text=True,
        timeout=RENDER_TIMEOUT,
        env=env,
        cwd=str(workspace),
    )
    log_path = workspace / f"{scene_file.stem}_log.txt"
    log_path.write_text(
        f"=== STDOUT ===\n{result.stdout}\n\n=== STDERR ===\n{result.stderr}",
        encoding="utf-8",
    )
    if result.returncode != 0:
        return None, result.stderr[-2000:] if result.stderr else "Unknown render error"
    videos_dir = media_dir / "videos"
    if not videos_dir.exists():
        return None, "Render succeeded but no videos/ dir found"
    matches = list(videos_dir.rglob(f"{scene_class}.mp4"))
    # Each round writes a new scene file but reuses the scene class, so old
    # rounds leave stale <class>.mp4 under their own <file_stem>/ dir. Scope to
    # THIS scene file's dir, and break any remaining tie by newest mtime.
    scoped = [m for m in matches if scene_file.stem in m.parts]
    if scoped:
        matches = scoped
    if not matches:
        matches = list(videos_dir.rglob("*.mp4"))
    if not matches:
        return None, "Render succeeded but no .mp4 found"
    return max(matches, key=lambda p: p.stat().st_mtime), None


# ── Rendering: remotion ───────────────────────────────────────────────


def ensure_remotion_project():
    proj = REMOTION_PROJECT_DIR
    if (proj / "node_modules").exists():
        return proj
    proj.mkdir(parents=True, exist_ok=True)
    (proj / "src" / "clips").mkdir(parents=True, exist_ok=True)
    (proj / "output").mkdir(parents=True, exist_ok=True)

    package_json = {
        "name": "eduvid-remotion",
        "version": "1.0.0",
        "private": True,
        "scripts": {"render": "remotion render"},
        "dependencies": {
            "react": "^19.0.0",
            "react-dom": "^19.0.0",
            "remotion": "^4.0.0",
            "@remotion/cli": "^4.0.0",
            "@remotion/google-fonts": "^4.0.0",
        },
        "devDependencies": {"typescript": "^5.0.0", "@types/react": "^19.0.0"},
    }
    save_json(package_json, proj / "package.json")

    tsconfig = {
        "compilerOptions": {
            "target": "ES2020",
            "lib": ["DOM", "DOM.Iterable"],
            "module": "ESNext",
            "jsx": "react-jsx",
            "strict": True,
            "moduleResolution": "bundler",
            "esModuleInterop": True,
            "skipLibCheck": True,
        },
        "include": ["src"],
    }
    save_json(tsconfig, proj / "tsconfig.json")

    (proj / "src" / "theme.ts").write_text(
        THEME_TS.read_text(encoding="utf-8"), encoding="utf-8"
    )

    index_ts = '''import { registerRoot } from "remotion";
import { loadFont } from "@remotion/google-fonts/Inter";
import { RemotionRoot } from "./Root";

loadFont();
registerRoot(RemotionRoot);
'''
    (proj / "src" / "index.ts").write_text(index_ts, encoding="utf-8")

    root_tsx = '''import { Composition } from "remotion";

export const RemotionRoot: React.FC = () => {
  return <></>;
};
'''
    (proj / "src" / "Root.tsx").write_text(root_tsx, encoding="utf-8")

    print("Installing remotion dependencies (one-time)...", file=sys.stderr)
    subprocess.run(
        ["npm", "install"], cwd=str(proj), capture_output=True, text=True, check=True
    )
    print("Remotion project ready.", file=sys.stderr)
    return proj


def regenerate_root_tsx(proj, comp_id, component_name, clip_file_rel, duration_frames):
    root_content = f'''import {{ Composition }} from "remotion";
import {{ {component_name} }} from "./{clip_file_rel}";

export const RemotionRoot: React.FC = () => {{
  return (
    <Composition
      id="{comp_id}"
      component={{{component_name}}}
      durationInFrames={{{duration_frames}}}
      fps={{{FPS}}}
      width={{{RESOLUTION_W}}}
      height={{{RESOLUTION_H}}}
    />
  );
}};
'''
    (proj / "src" / "Root.tsx").write_text(root_content, encoding="utf-8")


def render_remotion(proj, comp_id, output_path, log_prefix, workspace):
    result = subprocess.run(
        [
            "npx",
            "remotion",
            "render",
            comp_id,
            str(output_path),
            "--concurrency=2",
        ],
        capture_output=True,
        text=True,
        timeout=RENDER_TIMEOUT,
        cwd=str(proj),
    )
    log_path = workspace / f"{log_prefix}_log.txt"
    log_path.write_text(
        f"=== STDOUT ===\n{result.stdout}\n\n=== STDERR ===\n{result.stderr}",
        encoding="utf-8",
    )
    if result.returncode != 0 or not output_path.exists():
        return None, result.stderr[-2000:] if result.stderr else "Unknown render error"
    return output_path, None


# ── ffprobe ───────────────────────────────────────────────────────────


def ffprobe_duration(path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    try:
        return float(result.stdout.strip())
    except (ValueError, AttributeError):
        return None


# ── Seedance (OpenRouter) ─────────────────────────────────────────────

# Style anchors for cast-LESS seedance shots. Cast clips already carry the
# frozen look through their cast reference images; generic shots (crowds,
# animals, plain scenes) have no ref, so we inject the art-style text + these
# global anchor images (already rendered in the film's style) as visual
# guidance. Plain-language reference, per OpenRouter's docs — no @Image tags.
STYLE_ANCHOR_IMAGES = [
    CASTS_DIR / "ref" / "images" / "character_scientist_design.jpg",
    CASTS_DIR / "ref" / "images" / "environment_city_example.jpg",
]
ART_STYLE_PATH = PROMPTS_DIR / "art_style.json"


def _art_style_text():
    """Frozen art-style description (image style + video style) for prompts."""
    try:
        d = load_json(ART_STYLE_PATH)
    except Exception:
        return ""
    parts = [d.get("image", {}).get("styles", ""), d.get("video", {}).get("style", "")]
    return "  ".join(p for p in parts if p)


def build_seedance_skill(client, prompt_cfg, kimi_cfg, clip, chaining):
    """Assemble the seedance prompt-craft skill for this clip: always-on
    directing core + LLM-picked references + (when the clip chains) the
    first/last-frame continuity docs."""
    base = core_content("seedance")
    doc_index = build_doc_index("seedance")
    picked = []
    try:
        picked = llm_pick_docs(client, prompt_cfg, kimi_cfg, clip, doc_index, [], None)
    except Exception as e:  # noqa: BLE001
        print(f"  [seedance doc pick failed: {e}; core only]", file=sys.stderr)
    picked = [d for d in picked if any(x["name"] == d for x in doc_index)]
    if chaining:
        for d in SEEDANCE_CONTINUITY_DOCS:
            if d not in picked:
                picked.append(d)
    extra = load_picked_content("seedance", picked)
    if picked:
        print(f"  [seedance skill: core + {', '.join(picked)}]", file=sys.stderr)
    return base + ("\n\n" + extra if extra else "")


def llm_seedance_prompts(client, prompt_cfg, kimi_cfg, clip, segments, skill_content):
    """Craft the final seedance prompt(s) with the seedance skill loaded.

    Runs for EVERY clip (single- and multi-segment): turns the stage-8 shot
    intent + narration into directed, anti-slop prompts. Multi-segment → one
    batched call so the segments read as one continuous take. Returns a list
    aligned to `segments`; falls back to the main prompt on any failure.
    """
    design = clip.get("visual_design", {})
    main_prompt = design.get("prompt", clip.get("description", ""))
    tmpl = prompt_cfg.get("seedance_prompt")
    if not tmpl:
        return [main_prompt] * len(segments)

    seg_lines = []
    for i, seg in enumerate(segments, 1):
        dur = (seg["end_ms"] - seg["start_ms"]) / 1000.0
        tag = "（承接上一段，首帧将由上一段末帧给定）" if seg.get("continuation") else ""
        seg_lines.append(
            f"  第{i}段（约{dur:.1f}秒{tag}）：旁白「{seg.get('narration', '')}」"
        )

    task = tmpl["task"].format(
        skill=skill_content,
        main_prompt=main_prompt,
        cast="、".join(clip.get("cast_ids", [])) or "（无）",
        n=len(segments),
        segments="\n".join(seg_lines),
    )
    messages = [
        {"role": "system", "content": tmpl["system"]},
        {"role": "user", "content": task},
    ]
    try:
        raw = llm_call(client, messages, kimi_cfg, json_mode=True)
        prompts = json.loads(raw).get("segments", [])
    except Exception as e:  # noqa: BLE001
        print(f"  [seedance prompt gen failed: {e}; using main prompt]", file=sys.stderr)
        return [main_prompt] * len(segments)
    if len(prompts) != len(segments):
        print("  [seedance prompt count mismatch; using main prompt]", file=sys.stderr)
        return [main_prompt] * len(segments)
    return [p if isinstance(p, str) and p.strip() else main_prompt for p in prompts]


def _submit_seedance_segment(
    clip, segment_duration_sec, output_path, prompt_text, prev_last_frame_b64=None
):
    cfg = load_json(PROMPTS_DIR / "9_visual_flow_to_clips.json")["openrouter_cfg"]
    api_key = os.environ.get(cfg["api_key_env"], "")
    if not api_key:
        print(f"Error: {cfg['api_key_env']} not set", file=sys.stderr)
        return None, "No OpenRouter API key"

    # seedance-2.0 supported_durations = [4..15]; requesting 2-3s is rejected.
    # ceil (overshoot) so the raw clip is always >= target; render_seedance then
    # cuts the segment down to its exact frame count after download.
    duration = max(4, min(math.ceil(segment_duration_sec), 15))

    cast_imgs = [img for cid in clip.get("cast_ids", []) for img in find_cast_images(cid)]
    if cast_imgs:
        # Cast clips: the cast reference images already carry the frozen style.
        ref_imgs = cast_imgs
        final_prompt = prompt_text
    else:
        # Cast-less clips have no ref to anchor the look → inject the frozen
        # art-style text + the global style-anchor images so generic figures
        # and scenes still match the film's style.
        ref_imgs = [p for p in STYLE_ANCHOR_IMAGES if p.exists()]
        style_text = _art_style_text()
        # The anchors are STYLE references, not subjects. OpenRouter input_references
        # are generic "visual guidance" — the prompt must say what each is for, or
        # seedance inserts the reference's people/objects as literal content.
        final_prompt = (
            f"【参考图仅用于统一美术与渲染风格】——画风、材质质感、配色、造型语言："
            f"{style_text}\n"
            f"参考图里的人物、物体、场景【绝不要出现在画面中】，它们只示意风格、不是本镜头的内容；"
            f"本镜头实际画的东西，完全以下面的描述为准：\n\n{prompt_text}"
            if style_text
            else prompt_text
        )

    payload = {
        "model": cfg["video_model"],
        "prompt": final_prompt,
        "duration": duration,
        "resolution": "720p",
        "aspect_ratio": "16:9",
        # We mux our own narration in stage 10 — never want seedance's native
        # audio track (it also forced the stage-10 concat normalization).
        "generate_audio": False,
    }

    ref_refs = images_to_b64(ref_imgs)
    if ref_refs:
        payload["input_references"] = ref_refs

    if prev_last_frame_b64:
        payload["frame_images"] = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{prev_last_frame_b64}"},
                "frame_type": "first_frame",
            }
        ]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    base = cfg["api_base"]

    try:
        resp = requests.post(f"{base}/videos", headers=headers, json=payload, timeout=60)
        result = resp.json()
    except Exception as e:
        return None, f"Submit failed: {e}"

    if resp.status_code not in (200, 202):
        return None, f"Submit error {resp.status_code}: {result}"

    job_id = result.get("id")
    if not job_id:
        return None, f"No job_id in response: {result}"

    print(
        f"  Seedance job {job_id} submitted (duration={duration}s), polling...",
        file=sys.stderr,
    )
    max_polls = cfg.get("max_poll_time", 600) // cfg.get("poll_interval", 10)
    last_status = None
    for _ in range(max_polls):
        time.sleep(cfg.get("poll_interval", 10))
        try:
            poll = requests.get(
                f"{base}/videos/{job_id}", headers=headers, timeout=30
            )
            status_data = poll.json()
        except Exception as e:
            print(f"  [poll error: {e}]", file=sys.stderr)
            continue

        status = status_data.get("status", "")
        if status != last_status:  # only announce on change, not every poll
            print(f"  [{status}]", file=sys.stderr)
            last_status = status

        if status == "completed":
            urls = status_data.get("unsigned_urls", [])
            if not urls:
                return None, "Completed but no video URL"
            # unsigned_urls need the bearer token passed on download too
            dl_resp = requests.get(
                urls[0],
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=120,
            )
            if dl_resp.status_code != 200:
                return None, f"Download HTTP {dl_resp.status_code}: {dl_resp.text[:200]}"
            video_data = dl_resp.content
            ctype = dl_resp.headers.get("Content-Type", "")
            # Sanity: a real mp4 has the 'ftyp' box type at bytes 4:8 (the first
            # 4 bytes are the box SIZE, which varies — so don't test those).
            # Reject tiny JSON/HTML error payloads.
            if len(video_data) < 1024 or video_data[4:8] != b"ftyp":
                return None, (
                    f"Downloaded payload is not an mp4 ({len(video_data)} bytes, "
                    f"content-type={ctype!r}, starts with {video_data[:32]!r})"
                )
            output_path.write_bytes(video_data)
            return output_path, None
        elif status in ("failed", "cancelled", "expired"):
            return None, f"Seedance {status}: {status_data.get('error', 'unknown')}"

    return None, "Seedance polling timed out"


def _concat_videos(segment_paths, output_path):
    if len(segment_paths) == 1:
        if segment_paths[0].resolve() != output_path.resolve():
            output_path.write_bytes(segment_paths[0].read_bytes())
        return output_path

    concat_list = output_path.parent / "concat_list.txt"
    concat_list.write_text(
        "\n".join(f"file '{p.resolve()}'" for p in segment_paths) + "\n"
    )
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_list),
            "-c", "copy",
            str(output_path),
        ],
        capture_output=True,
        timeout=120,
    )
    if result.returncode != 0:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_list),
                "-c:v", "libx264",
                "-preset", "fast",
                "-pix_fmt", "yuv420p",
                str(output_path),
            ],
            capture_output=True,
            timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg concat failed: {result.stderr.decode()[-500:]}"
            )
    return output_path


def render_seedance(clip, workspace, client, prompt_cfg, kimi_cfg, prev_last_frame_b64=None):
    # Step 8 splits long seedance clips into <=15s segments with continuation
    # flags. Each segment after the first uses the previous segment's last
    # frame as its first frame; outputs are concatenated to form the clip.
    segments = clip.get("segments")
    if not segments:
        segments = [
            {
                "start_ms": clip.get("start_ms", 0),
                "end_ms": clip.get(
                    "end_ms",
                    clip.get("duration_ms", 0)
                    or int(clip_target_duration(clip) * 1000),
                ),
                "continuation": False,
            }
        ]

    # Load the seedance prompt-craft skill, then craft the final prompt(s) with
    # it. chaining = multi-segment or seeded from a previous clip's last frame.
    chaining = len(segments) > 1 or bool(prev_last_frame_b64)
    skill_content = build_seedance_skill(client, prompt_cfg, kimi_cfg, clip, chaining)
    seg_prompts = llm_seedance_prompts(
        client, prompt_cfg, kimi_cfg, clip, segments, skill_content
    )

    final = workspace / "clip.mp4"
    one_segment = len(segments) == 1
    segment_paths = []
    prev_frame = prev_last_frame_b64

    for i, seg in enumerate(segments):
        seg_duration = (seg["end_ms"] - seg["start_ms"]) / 1000.0
        if seg_duration <= 0:
            continue
        # Single segment → render straight to clip.mp4 (no redundant seg_0.mp4
        # that just gets byte-copied). Multi-segment uses seg_* for the concat.
        seg_output = final if one_segment else workspace / f"seg_{i}.mp4"
        tag = ", continuation" if seg.get("continuation") else ""
        print(
            f"  Segment {i + 1}/{len(segments)} ({seg_duration:.2f}s{tag})",
            file=sys.stderr,
        )

        video, error = _submit_seedance_segment(
            clip, seg_duration, seg_output, seg_prompts[i], prev_frame
        )
        if not video:
            return None, f"Segment {i + 1} failed: {error}"

        # The request was overshot (ceil); cut this segment to its EXACT target
        # frame count with a re-encode — frame-accurate, unlike -c copy. Because
        # we overshot, this only ever trims (real motion), never freeze-pads.
        frame_snap(video, max(1, round(seg_duration * FPS)), video)
        segment_paths.append(video)
        # seed the next segment from the CUT boundary, not the overshot tail
        prev_frame = extract_last_frame_b64(video)

    if not segment_paths:
        return None, "No segments rendered"

    # Single segment is already at clip.mp4; _concat_videos no-ops it. Multi
    # segment concatenates seg_* into clip.mp4.
    try:
        _concat_videos(segment_paths, final)
    except Exception as e:
        return None, f"Concat failed: {e}"

    return final, None


def extract_last_frame_b64(video_path):
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-sseof",
                "-0.1",
                "-i",
                str(video_path),
                "-update",
                "1",
                "-q:v",
                "1",
                "-f",
                "image2pipe",
                "-vcodec",
                "png",
                "pipe:1",
            ],
            capture_output=True,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout:
            return base64.b64encode(result.stdout).decode()
    except Exception:
        pass
    return None


# ── Failure handling ──────────────────────────────────────────────────


def trim_to_duration(input_path, target_duration, output_path):
    # ffmpeg can't read and overwrite the same file — if in and out paths are
    # the same, trim to a temp file then move it into place.
    input_path = Path(input_path)
    output_path = Path(output_path)
    in_place = input_path.resolve() == output_path.resolve()
    dest = output_path.with_suffix(".trim.mp4") if in_place else output_path
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-t",
            str(target_duration),
            "-c",
            "copy",
            str(dest),
        ],
        capture_output=True,
        timeout=60,
    )
    if in_place and dest.exists():
        dest.replace(output_path)


def frame_snap(input_path, target_frames, output_path):
    """Force a rendered clip to EXACTLY target_frames at FPS. Clone-pads the
    tail then caps at target_frames, so one path both trims (long) and freeze-
    pads (short). `-frames:v` short-circuits the filter, so the long pad is only
    generated as far as needed. Re-encodes — frame-accurate, unlike -c copy.
    Also forces FPS, so a stray non-24fps render can't slip through."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    in_place = input_path.resolve() == output_path.resolve()
    dest = output_path.with_suffix(".snap.mp4") if in_place else output_path
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(input_path),
            "-vf", "tpad=stop_mode=clone:stop_duration=60",
            "-frames:v", str(target_frames),
            "-r", str(FPS),
            "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
            str(dest),
        ],
        capture_output=True,
        timeout=120,
    )
    if in_place and dest.exists():
        dest.replace(output_path)


def write_black_video(duration, output_path):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s={RESOLUTION_W}x{RESOLUTION_H}:r={FPS}:d={duration}",
            "-c:v",
            "libx264",
            str(output_path),
        ],
        capture_output=True,
        timeout=60,
    )


# ── Scene/component naming ────────────────────────────────────────────


def clip_num(clip_id):
    digits = re.sub(r"[^0-9]", "", str(clip_id))
    return digits or "0"


def safe_name(clip_id):
    return re.sub(r"[^a-zA-Z0-9_]", "_", str(clip_id))


def remotion_comp_id(clip_id):
    """Remotion composition ids allow only [a-zA-Z0-9-] (and CJK) — no
    underscores. Hyphenate the id, and drop any leading clip prefix so we get
    a clean 'clip-2' rather than a doubled 'clip-clip-2'."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", str(clip_id)).strip("-")
    slug = re.sub(r"^clip-?", "", slug)  # strip existing clip/clip- prefix
    return f"clip-{slug}" if slug else "clip"


# ── Code-gen loop ─────────────────────────────────────────────────────


def process_code_clip(client, prompt_cfg, kimi_cfg, clip, clips, backend, chapter):
    clip_id = clip["clip_id"]
    cid_safe = safe_name(clip_id)
    num = clip_num(clip_id)
    workspace = TMP_DIR / "clips" / f"chapter_{chapter}" / cid_safe
    workspace.mkdir(parents=True, exist_ok=True)

    duration = clip_target_duration(clip)
    convert_to_clip_relative(clip)

    base_content = core_content(backend)        # always-on best-practices core
    doc_index = build_doc_index(backend)        # dynamic rules pool
    picked_rules = []
    code = None
    error = None
    last_video = None
    last_round = -1

    component_name = f"Clip{num}"
    scene_class = f"Clip{num}"

    for round_num in range(MAX_ROUNDS):
        last_round = round_num
        print(f"\n--- Clip {clip_id} ({backend}) round {round_num} ---", file=sys.stderr)

        # Phase 1: always-on core + dynamically pick extra rules
        new_rules = llm_pick_docs(
            client, prompt_cfg, kimi_cfg, clip, doc_index, picked_rules, error
        )
        for r in new_rules:
            if r not in picked_rules and any(d["name"] == r for d in doc_index):
                picked_rules.append(r)
        skill_content = base_content
        picked_extra = load_picked_content(backend, picked_rules)
        if picked_extra:
            skill_content += "\n\n" + picked_extra

        # Phase 2: generate or fix. Generate whenever we have no code yet —
        # i.e. the first round, or after an earlier generation failed to
        # produce any (so we retry generation instead of "fixing" None).
        if code is None:
            code = llm_generate(
                client, prompt_cfg, kimi_cfg, clip, clips, skill_content, backend, num
            )
            if not code:
                error = "Code extraction failed"
                continue
        else:
            edits_raw = llm_fix(
                client, prompt_cfg, kimi_cfg, clip, clips, code, error, skill_content, backend, num
            )
            edits = parse_edits(edits_raw)
            if not edits:
                error = "No edits parsed from fix response"
                continue
            code, applied = apply_edits(code, edits)
            if applied == 0:
                error = "No edits could be applied (SEARCH text not found)"
                continue

        # Write code
        lang = "python" if backend == "manim" else "typescript"
        ext = ".py" if backend == "manim" else ".tsx"
        if backend == "manim":
            code = sanitize_manim_code(code)  # kill LLM-added config.frame_rate
        code_file = workspace / f"{cid_safe}_v{round_num}{ext}"
        code_file.write_text(code, encoding="utf-8")

        # Phase 3: render
        if backend == "manim":
            code_file_abs = code_file
            video, render_error = render_manim(code_file_abs, scene_class, workspace)
        else:
            proj = ensure_remotion_project()
            clip_dir_rel = f"clips/chapter_{chapter}"
            (proj / "src" / clip_dir_rel).mkdir(parents=True, exist_ok=True)
            # Clips live in src/clips/chapter_N/ and import "../theme", which
            # resolves to src/clips/theme — so the frozen theme must sit there.
            (proj / "src" / "clips" / "theme.ts").write_text(
                THEME_TS.read_text(encoding="utf-8"), encoding="utf-8"
            )
            clip_file = proj / "src" / clip_dir_rel / f"{cid_safe}_v{round_num}.tsx"
            clip_file.write_text(code, encoding="utf-8")
            clip_file_rel = f"{clip_dir_rel}/{cid_safe}_v{round_num}"
            duration_frames = max(1, round(duration * FPS))
            comp_id = remotion_comp_id(clip_id)
            regenerate_root_tsx(proj, comp_id, component_name, clip_file_rel, duration_frames)
            output_path = workspace / f"{cid_safe}_v{round_num}.mp4"
            video, render_error = render_remotion(
                proj, comp_id, output_path, f"{cid_safe}_v{round_num}", workspace
            )

        if render_error:
            error = render_error
            continue

        last_video = video

        # Phase 3.5: duration check — frame-aware (within half a frame)
        actual = ffprobe_duration(video)
        if actual is None:
            error = "ffprobe could not read duration"
            continue

        # Frame-snap acceptance. Manim quantizes each animation to whole frames,
        # so the total lands a frame or two off a non-frame-aligned target.
        # Rather than fail and burn the fix loop chasing a sub-frame-exact
        # duration the LLM can't hit, accept anything within SNAP_WINDOW_S and
        # force it to EXACTLY target_frames (trim if long, freeze-pad if short).
        target_frames = max(1, round(duration * FPS))
        if abs(actual - duration) <= SNAP_WINDOW_S:
            final = workspace / "clip.mp4"
            if abs(actual - duration) < (0.5 / FPS):
                # already frame-exact (remotion always; lucky manim) — no re-encode
                if video != final:
                    import shutil

                    shutil.copy2(str(video), str(final))
                snapped = actual
            else:
                # off by a frame or two (typical manim) — snap to exact frames
                frame_snap(video, target_frames, final)
                snapped = ffprobe_duration(final) or (target_frames / FPS)
            print(f"  ✓ Clip {clip_id} OK (round {round_num + 1}, "
                  f"{actual:.2f}s → {target_frames}f/{snapped:.2f}s)", file=sys.stderr)
            return {
                "clip_id": clip_id,
                "method": backend,
                "status": "ok",
                "rounds_used": round_num + 1,
                "final_path": str(final),
                "actual_duration": round(snapped, 3),
                "target_duration": duration,
                "skills_picked": picked_rules,
            }

        error = f"Duration off by {actual - duration:+.2f}s (got {actual:.2f}s, need {duration:.2f}s)"

    # All rounds exhausted — recover from best attempt
    print(f"  ✗ Clip {clip_id} failed after {MAX_ROUNDS} rounds", file=sys.stderr)
    final = workspace / "clip.mp4"

    if last_video and last_video.exists():
        actual = ffprobe_duration(last_video) or duration
        # Snap the best attempt to exact length regardless of over/under, so the
        # clip never injects a gap at assembly. Label reflects long vs short.
        frame_snap(last_video, max(1, round(duration * FPS)), final)
        status = "trimmed" if actual > duration else "short"
        print(f"  → recovered as {status} ({actual:.2f}s → {duration}s)", file=sys.stderr)
    else:
        write_black_video(duration, final)
        status = "black"
        print(f"  → black video ({duration}s)", file=sys.stderr)

    return {
        "clip_id": clip_id,
        "method": backend,
        "status": status,
        "rounds_used": MAX_ROUNDS,
        "final_path": str(final),
        "actual_duration": round(ffprobe_duration(final) or duration, 3),
        "target_duration": duration,
        "skills_picked": picked_rules,
        "last_error": error,
    }


# ── LLM generate / fix ────────────────────────────────────────────────


def llm_generate(client, prompt_cfg, kimi_cfg, clip, clips, skill_content, backend, num):
    key = f"generate_{backend}"
    template = prompt_cfg[key]
    chapter_ctx = build_chapter_context(clips, clip["clip_id"])

    duration = clip_target_duration(clip)
    duration_frames = max(1, round(duration * FPS))
    task = template["task"].format(
        chapter_context=chapter_ctx,
        clip_id=clip["clip_id"],
        implementation=clip.get("description", ""),
        duration=duration,
        duration_frames=duration_frames,
        fps=FPS,
        beats=format_beats(clip.get("_rel_beats", [])),
        char_timings=format_char_timings(clip.get("_rel_char_timings", [])),
        skills=skill_content,
        clip_num=num,
    )

    messages = [
        {"role": "system", "content": template["system"]},
        {"role": "user", "content": task},
    ]

    lang = "python" if backend == "manim" else "typescript"
    raw = llm_call(client, messages, kimi_cfg)
    code = extract_code_with_retries(client, kimi_cfg, messages, raw, lang)
    return code


def llm_fix(client, prompt_cfg, kimi_cfg, clip, clips, code, error, skill_content, backend, num):
    key = f"fix_{backend}"
    template = prompt_cfg[key]
    chapter_ctx = build_chapter_context(clips, clip["clip_id"])

    duration = clip_target_duration(clip)
    duration_frames = max(1, round(duration * FPS))
    task = template["task"].format(
        chapter_context=chapter_ctx,
        clip_id=clip["clip_id"],
        implementation=clip.get("description", ""),
        duration=duration,
        duration_frames=duration_frames,
        fps=FPS,
        beats=format_beats(clip.get("_rel_beats", [])),
        code=code,
        error=error,
        skills=skill_content,
    )

    messages = [
        {"role": "system", "content": template["system"]},
        {"role": "user", "content": task},
    ]

    raw = llm_call(client, messages, kimi_cfg)
    return raw


# ── Main ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Step 9: Visual flow → clips")
    parser.add_argument("--chapter", type=int, required=True, help="Chapter number")
    parser.add_argument(
        "--clip-ids",
        default=None,
        help="Comma-separated clip IDs to process (default: all)",
    )
    parser.add_argument(
        "--prompt",
        default="prompts/9_visual_flow_to_clips.json",
        help="Path to prompt JSON",
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_DIR / ".env")

    # Load visual flow
    flow_path = TMP_DIR / "visual_flow" / f"visual_flow_chapter_{args.chapter}.json"
    if not flow_path.exists():
        print(f"Error: visual flow not found: {flow_path}", file=sys.stderr)
        print("Run step 8 first.", file=sys.stderr)
        sys.exit(1)

    flow = load_json(flow_path)
    clips = flow.get("clips", [])

    if args.clip_ids:
        wanted = {c.strip() for c in args.clip_ids.split(",")}
        clips = [c for c in clips if str(c.get("clip_id")) in wanted]

    print(f"Processing {len(clips)} clips for chapter {args.chapter}", file=sys.stderr)

    # Load prompt config
    prompt_cfg = load_json(args.prompt)
    kimi_cfg = prompt_cfg["kimi_cfg"]

    # Create LLM client
    client = make_client(kimi_cfg)

    # Process clips
    results = []
    prev_seedance_last_frame = None
    prev_was_seedance = False
    prev_cast_ids = []

    for clip in clips:
        clip_id = clip.get("clip_id", "?")
        method = clip.get("method", "?")
        print(f"\n{'=' * 60}", file=sys.stderr)
        print(f"Clip {clip_id} ({method})", file=sys.stderr)
        print(f"{'=' * 60}", file=sys.stderr)

        try:
            if method == "seedance":
                workspace = TMP_DIR / "clips" / f"chapter_{args.chapter}" / safe_name(clip_id)
                workspace.mkdir(parents=True, exist_ok=True)

                # Continuation: if prev clip was seedance with overlapping cast.
                # bool() because set & set is a set — without it `continuation`
                # gets recorded as a set and breaks the manifest JSON dump.
                use_continuation = bool(
                    prev_was_seedance
                    and prev_cast_ids
                    and set(clip.get("cast_ids", [])) & set(prev_cast_ids)
                )
                frame_b64 = prev_seedance_last_frame if use_continuation else None

                video, error = render_seedance(
                    clip, workspace, client, prompt_cfg, kimi_cfg, frame_b64
                )
                if video:
                    # Store last frame for potential continuation
                    prev_seedance_last_frame = extract_last_frame_b64(video)
                    prev_was_seedance = True
                    prev_cast_ids = clip.get("cast_ids", [])
                    # render_seedance cut each segment to its exact frame count,
                    # so the clip is already at target — no imprecise -c copy trim.
                    target = clip_target_duration(clip)
                    actual = ffprobe_duration(video) or target
                    results.append(
                        {
                            "clip_id": clip_id,
                            "method": "seedance",
                            "status": "ok",
                            "rounds_used": 0,
                            "final_path": str(video),
                            "actual_duration": round(actual, 3),
                            "target_duration": target,
                            "continuation": use_continuation,
                        }
                    )
                    print(f"  ✓ Seedance clip {clip_id} OK ({actual:.2f}s)", file=sys.stderr)
                else:
                    write_black_video(clip_target_duration(clip), workspace / "clip.mp4")
                    results.append(
                        {
                            "clip_id": clip_id,
                            "method": "seedance",
                            "status": "black",
                            "rounds_used": 0,
                            "final_path": str(workspace / "clip.mp4"),
                            "target_duration": clip_target_duration(clip),
                            "error": error,
                        }
                    )
                    prev_seedance_last_frame = None
                    prev_was_seedance = False
                    prev_cast_ids = []
                    print(f"  ✗ Seedance clip {clip_id} FAILED: {error}", file=sys.stderr)
            elif method in ("manim", "remotion"):
                result = process_code_clip(
                    client, prompt_cfg, kimi_cfg, clip, clips, method, args.chapter
                )
                results.append(result)
                prev_was_seedance = False
                prev_seedance_last_frame = None
                prev_cast_ids = []
            else:
                print(f"  Unknown method: {method}, skipping", file=sys.stderr)
                continue
        except Exception as e:
            import traceback

            print(f"  ✗✗ Clip {clip_id} CRASHED: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            workspace = TMP_DIR / "clips" / f"chapter_{args.chapter}" / safe_name(clip_id)
            workspace.mkdir(parents=True, exist_ok=True)
            write_black_video(clip_target_duration(clip), workspace / "clip.mp4")
            results.append(
                {
                    "clip_id": clip_id,
                    "method": method,
                    "status": "crashed",
                    "rounds_used": 0,
                    "final_path": str(workspace / "clip.mp4"),
                    "target_duration": clip_target_duration(clip),
                    "error": str(e),
                }
            )
            prev_was_seedance = False
            prev_seedance_last_frame = None
            prev_cast_ids = []

    # Save manifest
    manifest = {
        "chapter": args.chapter,
        "model": kimi_cfg["model"],
        "clip_count": len(results),
        "clips": results,
    }
    manifest_path = TMP_DIR / "clips" / f"chapter_{args.chapter}" / "clips_manifest.json"
    save_json(manifest, manifest_path)

    # Summary
    ok = sum(1 for r in results if r["status"] == "ok")
    trimmed = sum(1 for r in results if r["status"] == "trimmed")
    short = sum(1 for r in results if r["status"] == "short")
    black = sum(1 for r in results if r["status"] == "black")

    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"Done: {len(results)} clips", file=sys.stderr)
    print(f"  ok={ok}  trimmed={trimmed}  short={short}  black={black}", file=sys.stderr)
    print(f"Manifest: {manifest_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
