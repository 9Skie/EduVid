#!/usr/bin/env python3
"""
Stage 5: Extract cast roster from script, fetch real reference images, then describe.

3-phase pipeline:
  Phase 1: Extract  — DeepSeek reads script, outputs bare roster (no descriptions)
  Phase 2: Fetch    — Pure code: query Wikipedia/Wikimedia for 1 reference image per entity
  Phase 3: Describe — Single multimodal Kimi call: all entities + images → description cards
  Phase 4: Generate — GPT Image 2 produces stylized reference images from description cards
                      (portrait + design sheet for characters, scene for environments)

Usage (run from code/ with venv activated):
    source .venv/bin/activate
    python scripts/5_script_to_cast.py                     # full pipeline
    python scripts/5_script_to_cast.py --only extract      # phase 1
    python scripts/5_script_to_cast.py --only fetch        # phase 2
    python scripts/5_script_to_cast.py --only describe     # phase 3
    python scripts/5_script_to_cast.py --only generate     # phase 4
"""

import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import requests
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI


# --- Config ---

DEEPSEEK_CFG = {
    "model": "deepseek-v4-pro",
    "api_key_env": "DEEPSEEK_API_KEY",
    "api_base_env": "DEEPSEEK_API_BASE",
    "api_base_default": "https://api.deepseek.com/v1",
    "extra_body": {"thinking": {"type": "enabled"}},
    "reasoning_effort": "max",
    "max_tokens": 100000,
}

KIMI_CFG = {
    "model": "kimi-k2.6",
    "api_key_env": "KIMI_API_KEY",
    "api_base_env": "KIMI_API_BASE",
    "api_base_default": "https://api.moonshot.cn/v1",
    "extra_body": None,
    "max_tokens": 100000,
}

# Wikipedia API (Chinese)
WIKI_API = "https://zh.wikipedia.org/w/api.php"
WIKI_HEADERS = {"User-Agent": "EduVid/1.0 (educational video research; contact: dev@eduvid.local)"}
WIKI_THUMB_SIZE = 500       # request thumbnail at this max width
IMAGE_MAX_SIZE = 500        # resize longest edge to this
IMAGE_DPI = 100

# Image generation (OpenRouter)
OPENROUTER_IMAGE_API = "https://openrouter.ai/api/v1/images"
IMAGE_MODEL = "openai/gpt-image-2"
IMAGE_QUALITY = "high"
IMAGE_FORMAT = "png"
IMAGE_DPI = 100

# Per-type specs: (resolution tier, aspect ratio, exact pixel WxH)
PORTRAIT_SPEC = ("1K", "9:16", (720, 1280))
DESIGN_SPEC = ("2K", "21:9", (1680, 720))
ENVIRONMENT_SPEC = ("1K", "16:9", (1280, 720))

# Style reference images under casts/ref/images/
PORTRAIT_STYLE_REF = "character_scientist_example.jpg"
DESIGN_STYLE_REF = "character_scientist_design.jpg"
MODEL_SHEET_REF = "character_model_sheet_ref.jpg"
ENVIRONMENT_STYLE_REF = "environment_city_example.jpg"

DESIGN_SHEET_PROMPT = (
    "Create a character model sheet following the layout of the reference model sheet image. "
    "Show multiple views: front view, side view, back view, and close-up detail shots of key features. "
    "Keep the character design consistent with the provided character portrait. "
    "Clean white background, professional character design turnaround."
)


# --- Helpers ---

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def flatten_chapters(script_raw):
    """Convert chapter-structured JSON into readable text. Falls back to raw text."""
    try:
        obj = json.loads(script_raw)
        lines = []
        for ch in obj.get("chapters", []):
            lines.append(f"━━━ {ch['name']} ━━━")
            lines.append(ch.get("script", ""))
            lines.append("")
        return "\n".join(lines)
    except (json.JSONDecodeError, TypeError):
        return script_raw


def build_chapter_list(chapters):
    """Index chapters from the script (source of truth — matches what the
    LLM reads in {script} 1:1, so bucket assignment never gets misaligned)."""
    lines = []
    for i, ch in enumerate(chapters, 1):
        lines.append(f"  chapter_{i}：{ch.get('name', '?')}")
    return "\n".join(lines)


def stream_call(client, messages, cfg, use_json=False):
    """Stream one LLM call. Returns the full text content."""
    kwargs = dict(
        model=cfg["model"],
        messages=messages,
        max_tokens=cfg["max_tokens"],
        stream=True,
    )
    if use_json:
        kwargs["response_format"] = {"type": "json_object"}
    if cfg.get("reasoning_effort"):
        kwargs["reasoning_effort"] = cfg["reasoning_effort"]
    if cfg["extra_body"]:
        kwargs["extra_body"] = cfg["extra_body"]

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


def get_bucket_folder(bucket, casts_dir):
    """Resolve the folder path for a bucket type."""
    if bucket == "consistent":
        return casts_dir / "consistent"
    elif bucket.startswith("chapter_"):
        return casts_dir / "chapters" / bucket
    else:
        return casts_dir / bucket


# --- Phase 1: Extract ---

def run_extract(project_dir, casts_dir, tmp_dir, args):
    """Phase 1: DeepSeek reads script, extracts bare roster JSON, splits into files."""
    # Resolve script path
    script_path = Path(args.script) if args.script else tmp_dir / "script_final.json"
    if not script_path.is_absolute():
        script_path = project_dir / script_path
    if not script_path.exists():
        print(f"Error: script not found at {script_path}", file=sys.stderr)
        sys.exit(1)

    script_raw = script_path.read_text(encoding="utf-8")
    script_text = flatten_chapters(script_raw)
    print(f"Script: {script_path} ({len(script_text)} chars)", file=sys.stderr)

    # Build chapter list from the script itself (source of truth) so it
    # matches what the LLM reads in {script} exactly — 1:1, no drift.
    try:
        script_obj = json.loads(script_raw)
        chapter_list = build_chapter_list(script_obj.get("chapters", []))
    except json.JSONDecodeError:
        print("Error: script is not valid JSON; cannot build chapter list",
              file=sys.stderr)
        sys.exit(1)

    # Load prompt
    prompt_cfg = load_json(project_dir / args.prompt)
    extract_cfg = prompt_cfg["extract"]

    # Build system prompt
    task = (extract_cfg["task"]
            .replace("{chapter_list}", chapter_list)
            .replace("{script}", script_text))

    system_prompt = (
        f"{extract_cfg['identity']}\n\n"
        f"{extract_cfg['selection_criteria']}\n\n"
        f"{task}\n\n"
        f"{extract_cfg['output_format']}"
    )

    # Clean up old cast files before re-extraction
    import shutil
    for sub in ["consistent", "chapters"]:
        target = casts_dir / sub
        if target.exists():
            shutil.rmtree(target)
            print(f"Cleaned: {target}/", file=sys.stderr)

    # Build client
    cfg = KIMI_CFG
    api_key = os.environ.get(cfg["api_key_env"], "")
    api_base = os.environ.get(cfg["api_base_env"], cfg["api_base_default"])
    if not api_key:
        print(f"Error: {cfg['api_key_env']} not set in .env", file=sys.stderr)
        sys.exit(1)
    client = OpenAI(api_key=api_key, base_url=api_base, timeout=600)

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"PHASE 1: CAST EXTRACTION ({cfg['model']})", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"以下是教学视频脚本全文：\n\n{script_text}"},
    ]

    result = stream_call(client, messages, cfg, use_json=True)

    try:
        parsed = json.loads(result)
    except json.JSONDecodeError as e:
        print(f"\nError: JSON parse failed: {e}", file=sys.stderr)
        raw_path = tmp_dir / "cast_raw_output.txt"
        raw_path.write_text(result, encoding="utf-8")
        print(f"Raw output saved to {raw_path}", file=sys.stderr)
        sys.exit(1)

    roster = parsed.get("roster", [])
    print(f"\nExtracted {len(roster)} entities", file=sys.stderr)

    # Save full roster
    roster_path = tmp_dir / "cast_roster.json"
    roster_path.write_text(
        json.dumps({"roster": roster}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Roster saved: {roster_path}", file=sys.stderr)

    # Split into individual cast files (no description yet)
    counts = {}
    for entry in roster:
        bucket = entry.get("bucket", "consistent")
        folder = get_bucket_folder(bucket, casts_dir)
        folder.mkdir(parents=True, exist_ok=True)

        record = {
            "cast_id": entry["cast_id"],
            "name": entry["name"],
            "kind": entry["kind"],
            "bucket": bucket,
            "description": "",
        }
        out_path = folder / f"{entry['cast_id']}.json"
        out_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  {bucket}/{entry['cast_id']}.json — {entry['name']} ({entry['kind']})", file=sys.stderr)
        counts[bucket] = counts.get(bucket, 0) + 1

    print(f"\nDONE — {len(roster)} cast files written", file=sys.stderr)
    for bucket, count in sorted(counts.items()):
        print(f"  {bucket}/: {count}", file=sys.stderr)


# --- Phase 2: Fetch ---

def _wiki_page_image(title):
    """Query Wikipedia for the lead image thumbnail URL of an article.

    Returns the thumbnail URL string, or None if no image found.
    """
    params = {
        "action": "query",
        "titles": title,
        "prop": "pageimages",
        "piprop": "thumbnail",
        "pithumbsize": WIKI_THUMB_SIZE,
        "format": "json",
        "formatversion": 2,
    }
    resp = requests.get(WIKI_API, params=params, headers=WIKI_HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    pages = data.get("query", {}).get("pages", [])
    if not pages:
        return None

    page = pages[0]
    thumb = page.get("thumbnail", {}).get("source")
    return thumb


def _wiki_search_image(query):
    """Fallback: search Wikipedia for an article, then get its lead image.

    Returns (title, thumbnail_url) or (None, None).
    """
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": 1,
        "format": "json",
        "formatversion": 2,
    }
    resp = requests.get(WIKI_API, params=params, headers=WIKI_HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    results = data.get("query", {}).get("search", [])
    if not results:
        return None, None

    title = results[0]["title"]
    thumb = _wiki_page_image(title)
    return title, thumb


def _download_and_resize(url, out_path):
    """Download an image from URL, resize to max IMAGE_MAX_SIZE on longest edge at IMAGE_DPI."""
    resp = requests.get(url, headers=WIKI_HEADERS, timeout=60)
    resp.raise_for_status()

    img = Image.open(io.BytesIO(resp.content))
    img = img.convert("RGB")

    # Resize so longest edge = IMAGE_MAX_SIZE
    w, h = img.size
    if max(w, h) > IMAGE_MAX_SIZE:
        if w >= h:
            new_w = IMAGE_MAX_SIZE
            new_h = round(h * IMAGE_MAX_SIZE / w)
        else:
            new_h = IMAGE_MAX_SIZE
            new_w = round(w * IMAGE_MAX_SIZE / h)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    img.save(out_path, format="JPEG", quality=90, dpi=(IMAGE_DPI, IMAGE_DPI))


def run_fetch(project_dir, casts_dir, tmp_dir):
    """Phase 2: Fetch 1 Wikipedia reference image per entity. Pure code, no LLM."""
    roster_path = tmp_dir / "cast_roster.json"
    if not roster_path.exists():
        print(f"Error: {roster_path} not found. Run --only extract first.", file=sys.stderr)
        sys.exit(1)

    roster = load_json(roster_path).get("roster", [])
    if not roster:
        print("Error: roster is empty", file=sys.stderr)
        sys.exit(1)

    images_dir = tmp_dir / "cast_images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"PHASE 2: FETCH REFERENCE IMAGES (Wikipedia)", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"Entities: {len(roster)}", file=sys.stderr)

    manifest = {}
    found = 0
    missed = 0

    for entry in roster:
        cast_id = entry["cast_id"]
        name = entry["name"]
        out_path = images_dir / f"{cast_id}.jpg"

        # Skip if already downloaded
        if out_path.exists():
            print(f"  [{cast_id}] {name} — already cached", file=sys.stderr)
            manifest[cast_id] = out_path.name
            found += 1
            continue

        print(f"  [{cast_id}] {name} — searching...", end="", file=sys.stderr)

        try:
            # Try direct title lookup first
            thumb_url = _wiki_page_image(name)

            # Fallback: search
            resolved_title = None
            if not thumb_url:
                resolved_title, thumb_url = _wiki_search_image(name)

            if not thumb_url:
                print(f" MISS", file=sys.stderr)
                missed += 1
                continue

            _download_and_resize(thumb_url, out_path)
            label = f" (via '{resolved_title}')" if resolved_title else ""
            print(f" OK{label} ({out_path.stat().st_size // 1024}KB)", file=sys.stderr)
            manifest[cast_id] = out_path.name
            found += 1

        except Exception as e:
            print(f" ERROR: {e}", file=sys.stderr)
            missed += 1

    # Save manifest
    manifest_path = images_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nFetch done: {found} found, {missed} missed", file=sys.stderr)
    print(f"Manifest: {manifest_path}", file=sys.stderr)


# --- Phase 3: Describe ---

def _image_to_data_url(path):
    """Read a local image file and return a base64 data URL."""
    ext = Path(path).suffix.lower().lstrip(".")
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext or 'png'}"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def run_describe(project_dir, casts_dir, tmp_dir, args):
    """Phase 3: Single multimodal Kimi call. Images + roster + script → descriptions.

    Sends all entity images (interleaved with cast_id text labels) plus the full
    roster and script in one call. Kimi returns {cast_id, description} pairs;
    code merges them back into the roster and individual cast files.
    """
    # Load roster
    roster_path = tmp_dir / "cast_roster.json"
    if not roster_path.exists():
        print(f"Error: {roster_path} not found. Run --only extract first.", file=sys.stderr)
        sys.exit(1)
    roster = load_json(roster_path).get("roster", [])
    if not roster:
        print("Error: roster is empty", file=sys.stderr)
        sys.exit(1)

    # Load image manifest
    images_dir = tmp_dir / "cast_images"
    manifest_path = images_dir / "manifest.json"
    manifest = load_json(manifest_path) if manifest_path.exists() else {}

    # Load script
    script_path = Path(args.script) if args.script else tmp_dir / "script_final.json"
    if not script_path.is_absolute():
        script_path = project_dir / script_path
    if not script_path.exists():
        print(f"Error: script not found at {script_path}", file=sys.stderr)
        sys.exit(1)
    script_raw = script_path.read_text(encoding="utf-8")
    script_text = flatten_chapters(script_raw)

    # Load prompt
    prompt_cfg = load_json(project_dir / args.prompt)
    describe_cfg = prompt_cfg["describe"]

    # Fill placeholders in task text. The fixed anchors (teacher/students) are
    # NOT injected as exemplars anymore — their old cards described a speaking
    # classroom format and leaked that framing into every description.
    roster_text = json.dumps({"roster": roster}, ensure_ascii=False, indent=2)
    task_filled = (describe_cfg["task"]
                   .replace("{roster}", roster_text)
                   .replace("{script}", "（见下方消息）"))

    system_prompt = (
        f"{describe_cfg['identity']}\n\n"
        f"{task_filled}\n\n"
        f"{describe_cfg['output_format']}"
    )

    # --- Build multimodal content array ---
    # Interleave text labels with images so Kimi knows who each image depicts.
    content = [{"type": "text", "text": "以下是各实体的真实参考图片（按 cast_id 标注）："}]

    with_images = []
    without_images = []

    for entry in roster:
        cast_id = entry["cast_id"]
        name = entry["name"]
        filename = manifest.get(cast_id)

        if filename:
            img_path = images_dir / filename
            if img_path.exists():
                content.append({"type": "text", "text": f"--- {cast_id}（{name}）---"})
                content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(img_path)}})
                with_images.append(cast_id)
                continue

        without_images.append(f"{cast_id}（{name}）")

    if without_images:
        content.append({"type": "text", "text": f"以下实体无参考图片：{'、'.join(without_images)}"})

    content.append({"type": "text", "text": f"以下是完整脚本（供理解语境）：\n\n{script_text}"})
    content.append({"type": "text", "text": "请为名册中每个实体撰写视觉描述卡，输出 JSON。"})

    # Build Kimi client
    cfg = KIMI_CFG
    api_key = os.environ.get(cfg["api_key_env"], "")
    api_base = os.environ.get(cfg["api_base_env"], cfg["api_base_default"])
    if not api_key:
        print(f"Error: {cfg['api_key_env']} not set in .env", file=sys.stderr)
        sys.exit(1)
    client = OpenAI(api_key=api_key, base_url=api_base, timeout=600)

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"PHASE 3: DESCRIBE ({cfg['model']})", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"Total: {len(roster)} | With images: {len(with_images)} | Without: {len(without_images)}", file=sys.stderr)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]

    result = stream_call(client, messages, cfg, use_json=True)

    # Parse response: {"casts": [{"cast_id": "...", "description": "..."}]}
    try:
        parsed = json.loads(result)
    except json.JSONDecodeError as e:
        print(f"\nError: JSON parse failed: {e}", file=sys.stderr)
        raw_path = tmp_dir / "describe_raw_output.txt"
        raw_path.write_text(result, encoding="utf-8")
        print(f"Raw output saved to {raw_path}", file=sys.stderr)
        sys.exit(1)

    casts = parsed.get("casts", parsed.get("roster", []))
    desc_map = {item.get("cast_id"): item.get("description", "") for item in casts}

    # Merge descriptions back into roster and individual cast files
    updated = 0
    for entry in roster:
        cast_id = entry["cast_id"]
        desc = desc_map.get(cast_id, "")
        entry["description"] = desc

        bucket = entry.get("bucket", "consistent")
        folder = get_bucket_folder(bucket, casts_dir)
        cast_path = folder / f"{cast_id}.json"

        if cast_path.exists():
            record = load_json(cast_path)
            record["description"] = desc
            cast_path.write_text(
                json.dumps(record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        status = "OK" if desc else "EMPTY"
        print(f"  [{cast_id}] {entry['name']} — {status}", file=sys.stderr)
        if desc:
            updated += 1

    # Save updated roster with descriptions
    roster_path.write_text(
        json.dumps({"roster": roster}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nDONE — {updated}/{len(roster)} descriptions written", file=sys.stderr)
    print(f"Updated roster: {roster_path}", file=sys.stderr)


# --- Phase 4: Generate ---

def _load_art_style(project_dir):
    """Load the image art style prompt from art_style.json."""
    art_path = project_dir / "prompts" / "art_style.json"
    if not art_path.exists():
        return ""
    art_cfg = load_json(art_path)
    return art_cfg.get("image", {}).get("styles", "")


def call_image_api(api_key, prompt, input_references=None, resolution="1K",
                   aspect_ratio=None):
    """Call OpenRouter image API and return base64-encoded image bytes.

    Retries once on failure — the safety system occasionally rejects benign
    requests non-deterministically.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": IMAGE_MODEL,
        "prompt": prompt,
        "resolution": resolution,
        "quality": IMAGE_QUALITY,
        "output_format": IMAGE_FORMAT,
    }
    if aspect_ratio:
        payload["aspect_ratio"] = aspect_ratio
    if input_references:
        payload["input_references"] = input_references

    for attempt in (1, 2):
        try:
            response = requests.post(
                OPENROUTER_IMAGE_API, headers=headers, json=payload, timeout=300
            )
            response.raise_for_status()
            result = response.json()
            if not result.get("data"):
                raise RuntimeError(f"No image returned: {result}")
            cost = result.get("usage", {}).get("cost", "?")
            print(f"    Cost: ${cost}", file=sys.stderr)
            return result["data"][0]["b64_json"]
        except Exception as e:
            msg = str(e)
            resp = getattr(e, "response", None)
            if resp is not None:
                msg = f"HTTP {resp.status_code}: {resp.text[:300]}"
            print(f"    [attempt {attempt}/2] {msg}", file=sys.stderr)
            if attempt == 1:
                time.sleep(5)
                continue
            raise


try:
    _RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    _RESAMPLE = Image.LANCZOS


def _postprocess_image(image_path, target_size):
    """Resize image to exact pixel dimensions and set DPI metadata."""
    img = Image.open(image_path)
    img = img.convert("RGB")
    img = img.resize(target_size, _RESAMPLE)
    img.save(image_path, format="JPEG", quality=90, dpi=(IMAGE_DPI, IMAGE_DPI))


def _ref_data_url(ref_dir, filename):
    """Build a data URL for a style reference image, or None if missing."""
    p = ref_dir / filename
    if p.exists():
        return _image_to_data_url(p)
    return None


def _generate_character_images(api_key, art_style, cast_record, cast_path,
                               ref_dir, wiki_ref_path, project_dir):
    """Generate portrait + design sheet for a character.

    Portrait: wiki photo (likeness) + style ref (art direction)
    Design sheet: portrait + model sheet template + design style ref
    """
    cast_id = cast_record["cast_id"]
    description = cast_record.get("description", "")
    kind = "character"
    images_dir = cast_path.parent / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Portrait
    portrait_prompt = f"{art_style} {description}"
    print(f"  [{cast_id}] Generating portrait...", file=sys.stderr)

    portrait_refs = []
    if wiki_ref_path:
        portrait_refs.append(
            {"type": "image_url", "image_url": {"url": _image_to_data_url(wiki_ref_path)}}
        )
    style_url = _ref_data_url(ref_dir, PORTRAIT_STYLE_REF)
    if style_url:
        portrait_refs.append(
            {"type": "image_url", "image_url": {"url": style_url}}
        )

    portrait_b64 = call_image_api(
        api_key, portrait_prompt,
        input_references=portrait_refs or None,
        resolution=PORTRAIT_SPEC[0],
        aspect_ratio=PORTRAIT_SPEC[1],
    )
    portrait_path = images_dir / f"{kind}_{cast_id}_example.jpg"
    portrait_path.write_bytes(base64.b64decode(portrait_b64))
    _postprocess_image(portrait_path, PORTRAIT_SPEC[2])
    print(f"  [{cast_id}] Portrait saved: {portrait_path.name}", file=sys.stderr)

    # Step 2: Design sheet (skip if model sheet template missing)
    model_sheet_url = _ref_data_url(ref_dir, MODEL_SHEET_REF)
    if not model_sheet_url:
        print(f"  [{cast_id}] No model sheet template, skipping design sheet", file=sys.stderr)
        cast_record["reference_images"] = [str(portrait_path.relative_to(project_dir))]
        cast_path.write_text(
            json.dumps(cast_record, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        return

    time.sleep(1)
    design_prompt = f"{art_style} {DESIGN_SHEET_PROMPT}"
    design_refs = [
        {"type": "image_url", "image_url": {"url": _image_to_data_url(portrait_path)}},
        {"type": "image_url", "image_url": {"url": model_sheet_url}},
    ]
    design_style_url = _ref_data_url(ref_dir, DESIGN_STYLE_REF)
    if design_style_url:
        design_refs.append(
            {"type": "image_url", "image_url": {"url": design_style_url}}
        )

    print(f"  [{cast_id}] Generating design sheet...", file=sys.stderr)
    design_b64 = call_image_api(
        api_key, design_prompt,
        input_references=design_refs,
        resolution=DESIGN_SPEC[0],
        aspect_ratio=DESIGN_SPEC[1],
    )
    design_path = images_dir / f"{kind}_{cast_id}_design.jpg"
    design_path.write_bytes(base64.b64decode(design_b64))
    _postprocess_image(design_path, DESIGN_SPEC[2])
    print(f"  [{cast_id}] Design sheet saved: {design_path.name}", file=sys.stderr)

    cast_record["reference_images"] = [
        str(portrait_path.relative_to(project_dir)),
        str(design_path.relative_to(project_dir)),
    ]
    cast_path.write_text(
        json.dumps(cast_record, ensure_ascii=False, indent=2), encoding="utf-8",
    )


def _generate_environment_images(api_key, art_style, cast_record, cast_path,
                                 ref_dir, project_dir):
    """Generate a scene image for an environment, using style ref for art direction."""
    cast_id = cast_record["cast_id"]
    description = cast_record.get("description", "")
    kind = "environment"
    images_dir = cast_path.parent / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    scene_prompt = f"{art_style} {description}"

    env_refs = []
    style_url = _ref_data_url(ref_dir, ENVIRONMENT_STYLE_REF)
    if style_url:
        env_refs.append({"type": "image_url", "image_url": {"url": style_url}})

    print(f"  [{cast_id}] Generating scene...", file=sys.stderr)
    scene_b64 = call_image_api(
        api_key, scene_prompt,
        input_references=env_refs or None,
        resolution=ENVIRONMENT_SPEC[0],
        aspect_ratio=ENVIRONMENT_SPEC[1],
    )
    scene_path = images_dir / f"{kind}_{cast_id}_example.jpg"
    scene_path.write_bytes(base64.b64decode(scene_b64))
    _postprocess_image(scene_path, ENVIRONMENT_SPEC[2])
    print(f"  [{cast_id}] Scene saved: {scene_path.name}", file=sys.stderr)

    cast_record["reference_images"] = [str(scene_path.relative_to(project_dir))]
    cast_path.write_text(
        json.dumps(cast_record, ensure_ascii=False, indent=2), encoding="utf-8",
    )


def run_generate(project_dir, casts_dir, tmp_dir, args):
    """Phase 4: Generate stylized reference images from description cards.

    Characters get portrait + design sheet (two API calls each).
    Environments get a single scene image.
    Skips entries that already have reference_images populated.
    """
    roster_path = tmp_dir / "cast_roster.json"
    if not roster_path.exists():
        print(f"Error: {roster_path} not found. Run --only extract first.", file=sys.stderr)
        sys.exit(1)
    roster = load_json(roster_path).get("roster", [])
    if not roster:
        print("Error: roster is empty", file=sys.stderr)
        sys.exit(1)

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not set in .env", file=sys.stderr)
        sys.exit(1)

    art_style = _load_art_style(project_dir)
    print(f"Art style loaded: {len(art_style)} chars", file=sys.stderr)

    ref_dir = casts_dir / "ref" / "images"

    # Load wiki image manifest for portrait references
    wiki_manifest_path = tmp_dir / "cast_images" / "manifest.json"
    wiki_manifest = load_json(wiki_manifest_path) if wiki_manifest_path.exists() else {}

    # Separate characters and environments, resolve wiki ref paths
    characters = []
    environments = []

    for entry in roster:
        cast_id = entry["cast_id"]
        bucket = entry.get("bucket", "consistent")
        cast_path = get_bucket_folder(bucket, casts_dir) / f"{cast_id}.json"

        if not cast_path.exists():
            print(f"  [{cast_id}] Cast file not found, skipping", file=sys.stderr)
            continue

        cast_record = load_json(cast_path)

        if cast_record.get("reference_images"):
            print(f"  [{cast_id}] Already has images, skipping", file=sys.stderr)
            continue

        # Resolve wiki ref photo path
        wiki_ref_path = None
        filename = wiki_manifest.get(cast_id)
        if filename:
            p = tmp_dir / "cast_images" / filename
            if p.exists():
                wiki_ref_path = p

        if entry.get("kind") == "location":
            environments.append((cast_record, cast_path))
        else:
            characters.append((cast_record, cast_path, wiki_ref_path))

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"PHASE 4: GENERATE ({IMAGE_MODEL})", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"Characters: {len(characters)} | Environments: {len(environments)}", file=sys.stderr)

    # Characters first (all portraits, then design sheets)
    for cast_record, cast_path, wiki_ref_path in characters:
        try:
            _generate_character_images(
                api_key, art_style, cast_record, cast_path,
                ref_dir, wiki_ref_path, project_dir,
            )
        except Exception as e:
            print(f"  [{cast_record['cast_id']}] ERROR: {e}", file=sys.stderr)
        time.sleep(1)

    # Environments
    for cast_record, cast_path in environments:
        try:
            _generate_environment_images(
                api_key, art_style, cast_record, cast_path, ref_dir, project_dir
            )
        except Exception as e:
            print(f"  [{cast_record['cast_id']}] ERROR: {e}", file=sys.stderr)
        time.sleep(1)

    # Fixed anchors go through the same generation path → casts/fixed/images/
    generate_anchors(project_dir, casts_dir, api_key, art_style, ref_dir)

    print(f"\nDONE — image generation complete", file=sys.stderr)


# --- Anchors (teacher + two students) — generated as part of `generate` ---

# Persistent on-screen anchors. They are NOT extracted from the script and
# behave like consistent characters: the clip-plan LLM decides if/when they
# appear (rules live in the stage 7/8 prompts), and they never speak. The
# descriptions here are appearance-only.
ANCHOR_IDS = ["teacher", "xiaoming", "xiaohong"]

GENDER_WORD = {"male": "男", "female": "女"}

ANCHOR_DESC = {
    "teacher": (
        "一位成年中国{gw}教师，约三十多岁，亲切、有耐心、有权威感的神情，"
        "得体的日常教师着装。"
    ),
    "xiaoming": (
        "一名约{age}岁的中国男学生，符合该年龄段的身形与脸庞；短发，整洁的日常校园"
        "穿着，好奇、外向的神情。与小红外观明显区分（性别、发型、配色）。"
    ),
    "xiaohong": (
        "一名约{age}岁的中国女学生，符合该年龄段的身形与脸庞；扎起的头发或齐肩短发，"
        "整洁的日常校园穿着，细心、专注的神情。与小明外观明显区分（性别、发型、配色）。"
    ),
}


def _load_teacher_meta(project_dir):
    """Read grade_level / teacher_photo_path / teacher_gender from 1_teacher.json."""
    try:
        cfg = load_json(project_dir / "prompts" / "1_teacher.json")
    except Exception:
        cfg = {}
    try:
        grade = int(cfg.get("grade_level", 8))
    except (TypeError, ValueError):
        grade = 8
    grade = max(1, min(grade, 12))
    gender = cfg.get("teacher_gender", "male")
    if gender not in GENDER_WORD:
        gender = "male"
    return grade, (cfg.get("teacher_photo_path") or ""), gender


def grade_to_age(grade):
    """Grade 1–12 → approximate student age. Chinese grade 1 starts at age 6
    (义务教育法: 年满 6 周岁入学), so age ≈ grade + 5."""
    return grade + 5


def generate_anchors(project_dir, casts_dir, api_key, art_style, ref_dir):
    """(Re)generate the three fixed anchors. Runs at the end of `generate`, so
    they go through the same image-generation path as every other cast member.

    Teacher → stylized from the real photo (if provided), gendered by
    teacher_gender; students → from the grade-derived age. Always overwrites
    casts/fixed/<id>.json description + reference_images.
    """
    grade, photo_rel, teacher_gender = _load_teacher_meta(project_dir)
    age = grade_to_age(grade)
    gw = GENDER_WORD.get(teacher_gender, "男")
    fixed_dir = casts_dir / "fixed"

    teacher_photo = None
    if photo_rel:
        p = Path(photo_rel)
        if not p.is_absolute():
            p = project_dir / photo_rel
        if p.exists():
            teacher_photo = p

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"ANCHORS (grade={grade}, age≈{age}, teacher={teacher_gender})", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"Teacher photo: {teacher_photo or '（无 — 纯生成）'}", file=sys.stderr)

    for cast_id in ANCHOR_IDS:
        cast_path = fixed_dir / f"{cast_id}.json"
        if not cast_path.exists():
            print(f"  [{cast_id}] fixed card missing, skipping", file=sys.stderr)
            continue

        record = load_json(cast_path)
        if cast_id == "teacher":
            record["description"] = ANCHOR_DESC["teacher"].format(gw=gw)
        else:
            record["description"] = ANCHOR_DESC[cast_id].format(age=age)
        record["reference_images"] = []
        cast_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        wiki_ref = teacher_photo if cast_id == "teacher" else None
        try:
            _generate_character_images(
                api_key, art_style, record, cast_path,
                ref_dir, wiki_ref, project_dir,
            )
        except Exception as e:
            print(f"  [{cast_id}] ERROR: {e}", file=sys.stderr)
        time.sleep(1)


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Extract cast, fetch images, describe")
    parser.add_argument("--only",
                        choices=["extract", "fetch", "describe", "generate"],
                        default=None,
                        help="Run only one phase (anchors are part of generate)")
    parser.add_argument("--script", default=None,
                        help="Path to script JSON (default: tmp/script_final.json)")
    parser.add_argument("--prompt", default="prompts/5_cast.json",
                        help="Path to cast prompt JSON")
    args = parser.parse_args()

    project_dir = Path(__file__).parent.parent
    load_dotenv(project_dir / ".env")
    tmp_dir = project_dir / "tmp"
    casts_dir = project_dir / "casts"

    if args.only in (None, "extract"):
        run_extract(project_dir, casts_dir, tmp_dir, args)

    if args.only in (None, "fetch"):
        run_fetch(project_dir, casts_dir, tmp_dir)

    if args.only in (None, "describe"):
        run_describe(project_dir, casts_dir, tmp_dir, args)

    if args.only in (None, "generate"):
        run_generate(project_dir, casts_dir, tmp_dir, args)


if __name__ == "__main__":
    main()
