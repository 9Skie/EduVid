#!/usr/bin/env python3
"""
Step 2: cast establishment.

Input:
  projects/<project_id>/script_final.json
  projects/<project_id>/context_pack.json   # currently unused except for future context
  tools/casts_fixed/*.json                  # anchor regeneration specs
  tools/prompts/2_casts.json
  tools/prompts/art_style.json              # optional

Output:
  projects/<project_id>/casts/roster.json
  projects/<project_id>/casts/wiki_manifest.json
  projects/<project_id>/casts/descriptions.json
  projects/<project_id>/casts/cast_set.json
  projects/<project_id>/casts/images/*.jpg

Rules:
  - real-entity reference images come from Wikipedia API only;
  - if no usable Wikipedia image is found, continue and mark real_image_found=false;
  - no hard cast cap, but only recurring specific identifiable people/locations;
  - teacher/students/classroom anchors are regenerated per project before roster extraction.

Usage (run from code/ with venv activated):
    source .venv/bin/activate
    python scripts/2_casts.py --project-id triangles
"""

import argparse
import base64
import io
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image


DEFAULT_MODEL = os.environ.get("KIMI_MODEL", "moonshotai/kimi-k3")
WIKI_API = "https://zh.wikipedia.org/w/api.php"
WIKI_HOST = "zh.wikipedia.org"
WIKI_HEADERS = {"User-Agent": "EduVidCasts/1.0 (local educational video pipeline)"}
WIKI_THUMB_SIZE = 500
IMAGE_MAX_EDGE = 1024
OPENROUTER_IMAGE_API = "https://openrouter.ai/api/v1/images"
IMAGE_MODEL = "openai/gpt-image-2"
ANCHOR_IDS = {"teacher", "xiaoming", "xiaohong", "classroom"}
ANCHOR_NAMES = {"老师", "小明", "小红", "教室"}
MAX_WORKERS = 8  # concurrent wiki fetches / image generations in phases 2 and 4

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


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def rel_to_code(path, code_dir):
    path = Path(path).resolve()
    try:
        return path.relative_to(code_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def flatten_script(script_obj):
    parts = []
    for ch in script_obj.get("chapters", []) or []:
        if not isinstance(ch, dict):
            continue
        if ch.get("title"):
            parts.append(f"第{ch.get('chapter_id')}章 {ch['title']}")
        for line in ch.get("lines", []) or []:
            if isinstance(line, dict) and line.get("text"):
                parts.append(line["text"])
    return "\n".join(parts)


def build_chapter_list(chapters):
    lines = []
    for ch in chapters or []:
        if isinstance(ch, dict):
            lines.append(f"chapter_{ch.get('chapter_id')}: {ch.get('title', '')}")
    return "\n".join(lines)


def stream_llm_content(client, messages, model, label=""):
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
        raw = stream_llm_content(client, messages, model, label=f"{label} attempt {attempt}")
        messages.append({"role": "assistant", "content": raw})
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            last_errors = [f"JSON parse failed: {e}"]
            parsed = None
        if parsed is not None:
            last_errors = validate_fn(parsed)
            if not last_errors:
                return parsed
        messages.append(
            {
                "role": "user",
                "content": (
                    f"{label} validation failed. Return corrected pure JSON only.\n"
                    + json.dumps({"errors": last_errors}, ensure_ascii=False, indent=2)
                ),
            }
        )
    raise RuntimeError(f"{label} failed after retries: {last_errors}")


def image_to_data_url(path):
    ext = Path(path).suffix.lower().lstrip(".")
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext or 'png'}"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def resize_longest_edge(img, max_edge):
    w, h = img.size
    if max(w, h) <= max_edge:
        return img
    if w >= h:
        new_w = max_edge
        new_h = round(h * max_edge / w)
    else:
        new_h = max_edge
        new_w = round(w * max_edge / h)
    return img.resize((new_w, new_h), Image.LANCZOS)


def save_jpg(img, out_path, max_edge=IMAGE_MAX_EDGE):
    """Save a reference image with its native aspect ratio intact, long edge capped.

    These are identity references fed to GPT Image 2 and Seedance, not video frames --
    nothing downstream requires 16:9 (the 1280x720 gate in 4_compose.py applies to
    rendered clips). This used to ImageOps.fit() to 1280x720, which scales to fill and
    centre-crops: a full-body character sheet lost everything but a torso band, and a
    wide multi-pose design sheet lost its outer poses. Scale only, never crop.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = resize_longest_edge(img.convert("RGB"), max_edge)
    img.save(out_path, format="JPEG", quality=90, dpi=(100, 100))


def wiki_page_image(title):
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
    pages = resp.json().get("query", {}).get("pages", [])
    if not pages:
        return None
    page = pages[0]
    thumb = page.get("thumbnail", {}).get("source")
    if not thumb:
        return None
    pageid = page.get("pageid")
    return {
        "title": page.get("title"),
        "pageid": pageid,
        "thumb": thumb,
        "url": f"https://zh.wikipedia.org/?curid={pageid}" if pageid else None,
    }


def wiki_search_image(query):
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
    results = resp.json().get("query", {}).get("search", [])
    if not results:
        return None
    return wiki_page_image(results[0]["title"])


def fetch_wiki_image(name, out_path):
    info = wiki_page_image(name) or wiki_search_image(name)
    if not info:
        return {"real_image_found": False, "source": None, "wiki_image": None}
    resp = requests.get(info["thumb"], headers=WIKI_HEADERS, timeout=60)
    resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content))
    save_jpg(img, out_path)
    info = dict(info)
    info.pop("thumb", None)
    return {"real_image_found": True, "source": info, "wiki_image": out_path.name}


def call_image_api(api_key, prompt, input_references=None, label=""):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": IMAGE_MODEL,
        "prompt": prompt,
        "resolution": "1K",
        "quality": "high",
        "output_format": "png",
    }
    if input_references:
        payload["input_references"] = input_references

    for attempt in (1, 2):
        try:
            response = requests.post(OPENROUTER_IMAGE_API, headers=headers, json=payload, timeout=300)
            response.raise_for_status()
            result = response.json()
            if not result.get("data"):
                raise RuntimeError(f"No image returned: {result}")
            cost = result.get("usage", {}).get("cost", "?")
            print(f"    Cost: ${cost}", file=sys.stderr)
            trace_call({"kind": "image_call", "label": label, "model": IMAGE_MODEL, "prompt": prompt, "n_refs": len(input_references or []), "cost": cost})
            return base64.b64decode(result["data"][0]["b64_json"])
        except Exception as e:  # noqa: BLE001
            resp = getattr(e, "response", None)
            msg = f"HTTP {resp.status_code}: {resp.text[:300]}" if resp is not None else str(e)
            print(f"    [attempt {attempt}/2] {msg}", file=sys.stderr)
            if attempt == 1:
                time.sleep(5)
                continue
            raise


def chapter_buckets(chapter_ids):
    """The exact bucket vocabulary for this script: consistent, or one real chapter."""
    return ["consistent"] + [f"chapter_{cid}" for cid in chapter_ids]


def bucket_to_chapters(bucket, chapter_ids):
    """Map a roster bucket onto real chapter ids.

    Matches against the actual ids rather than parsing an int out of the bucket name,
    so this cannot silently mis-scope when ids are not what the regex expected.
    Unrecognised buckets fall back to every chapter, which is the safe direction: a
    cast member available everywhere can go unused, one wrongly scoped out cannot be drawn.
    """
    bucket = str(bucket or "").strip()
    if bucket == "consistent":
        return chapter_ids
    for cid in chapter_ids:
        if bucket == f"chapter_{cid}":
            return [cid]
    return chapter_ids


def validate_roster_factory(chapter_ids):
    valid_buckets = set(chapter_buckets(chapter_ids))

    def validate(parsed):
        errors = []
        roster = parsed.get("roster") if isinstance(parsed, dict) else None
        if not isinstance(roster, list):
            return ["output must be an object with roster list"]
        seen = set()
        for item in roster:
            if not isinstance(item, dict):
                errors.append("roster items must be objects")
                continue
            cast_id = item.get("cast_id", "")
            name = item.get("name", "")
            kind = item.get("kind", "")
            bucket = item.get("bucket", "")
            if not re.fullmatch(r"[a-z][a-z0-9_]*", cast_id or ""):
                errors.append(f"bad cast_id {cast_id!r}")
            if cast_id in seen:
                errors.append(f"duplicate cast_id {cast_id}")
            seen.add(cast_id)
            if cast_id in ANCHOR_IDS or name in ANCHOR_NAMES:
                errors.append(f"do not extract fixed anchor {cast_id or name}")
            if not name:
                errors.append(f"{cast_id} missing name")
            if kind not in ("character", "location"):
                errors.append(f"{cast_id} kind must be character or location")
            if bucket not in valid_buckets:
                errors.append(f"{cast_id} bucket {bucket!r} must be one of {sorted(valid_buckets)}")
        return errors
    return validate


def validate_descriptions_factory(roster):
    want = {item["cast_id"] for item in roster}

    def validate(parsed):
        errors = []
        casts = parsed.get("casts") if isinstance(parsed, dict) else None
        if not isinstance(casts, list):
            return ["output must be an object with casts list"]
        got = {}
        for item in casts:
            if not isinstance(item, dict):
                errors.append("casts items must be objects")
                continue
            cast_id = item.get("cast_id")
            if cast_id not in want:
                errors.append(f"description for unknown cast_id {cast_id}")
                continue
            if not item.get("description"):
                errors.append(f"{cast_id} empty description")
            got[cast_id] = item.get("description", "")
        missing = want - set(got)
        if missing:
            errors.append(f"missing descriptions for {sorted(missing)}")
        return errors
    return validate


def validate_cast_set(cast_set, code_dir):
    errors = []
    cast = cast_set.get("cast") if isinstance(cast_set, dict) else None
    if not isinstance(cast, list) or not cast:
        return ["cast_set.cast must be a non-empty list"]
    seen = set()
    anchor_count = 0
    for item in cast:
        if not isinstance(item, dict):
            errors.append("cast items must be objects")
            continue
        cast_id = item.get("cast_id")
        if not cast_id:
            errors.append("cast item missing cast_id")
            continue
        if cast_id in seen:
            errors.append(f"duplicate cast_id {cast_id}")
        seen.add(cast_id)
        if item.get("is_anchor"):
            anchor_count += 1
        for key in ("name", "kind", "description", "reference_images"):
            if key not in item:
                errors.append(f"{cast_id} missing {key}")
        if item.get("kind") not in ("character", "location"):
            errors.append(f"{cast_id} kind must be character or location")
        refs = item.get("reference_images") or []
        if not isinstance(refs, list) or not refs:
            errors.append(f"{cast_id} reference_images must be a non-empty list")
            continue
        for ref in refs:
            if not (code_dir / ref).exists():
                errors.append(f"{cast_id} reference image not found: {ref}")
    if anchor_count == 0:
        errors.append("cast_set needs at least one fixed anchor")
    required = {"teacher", "classroom"}
    missing_required = required - seen
    if missing_required:
        errors.append(f"cast_set missing required anchors: {sorted(missing_required)}")
    if not ({"xiaoming", "xiaohong"} & seen):
        errors.append("cast_set needs at least one student anchor (xiaoming or xiaohong)")
    return errors


def load_teacher_job(code_dir):
    return load_json(code_dir / "tools" / "prompts" / "1_preproduction.json")["teacher_job"]


def _image_refs(paths):
    return [{"type": "image_url", "image_url": {"url": image_to_data_url(p)}} for p in paths if p and Path(p).exists()]


def regenerate_anchors(code_dir, project_dir, art_style, openrouter_key):
    if not openrouter_key:
        raise RuntimeError("OPENROUTER_API_KEY not set; anchors must be regenerated")

    specs_dir = code_dir / "tools" / "casts_fixed"
    casts_dir = project_dir / "casts"
    images_dir = casts_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    job = load_teacher_job(code_dir)
    grade_level = int(job.get("grade_level", 8))

    # Each chain is independent; the two steps WITHIN a chain are dependent
    # (example -> design) and stay sequential. Chains run in parallel.

    def teacher_chain():
        # Teacher: real photo + example -> stylized example; stylized example + design -> design.
        teacher_spec = load_json(specs_dir / "teacher.json")
        regen = teacher_spec["regen"]
        base_example = code_dir / regen["base_example"]
        base_design = code_dir / regen["base_design"]
        photo_path = job.get("teacher_photo_path", "")
        teacher_photo = code_dir / photo_path if photo_path else None
        teacher_photo_used = bool(teacher_photo and teacher_photo.exists())

        example_out = images_dir / regen["output_example"]
        design_out = images_dir / regen["output_design"]
        step1_refs = [teacher_photo if teacher_photo_used else base_example, base_example]
        image_bytes = call_image_api(openrouter_key, regen["example_prompt"], input_references=_image_refs(step1_refs), label="anchor teacher example")
        save_jpg(Image.open(io.BytesIO(image_bytes)), example_out)
        image_bytes = call_image_api(openrouter_key, regen["design_prompt"], input_references=_image_refs([example_out, base_design]), label="anchor teacher design")
        save_jpg(Image.open(io.BytesIO(image_bytes)), design_out)
        return {
            "cast_id": teacher_spec["cast_id"],
            "is_anchor": True,
            "name": teacher_spec["name"],
            "kind": teacher_spec["kind"],
            "reference_images": [rel_to_code(example_out, code_dir), rel_to_code(design_out, code_dir)],
            "description": teacher_spec["description"],
            "presence": teacher_spec["presence"],
            "teacher_photo_used": teacher_photo_used,
        }

    # Students: grade -> age, age + example -> age-changed example, + design -> design.
    students_spec = load_json(specs_dir / "students.json")
    age = grade_level + 5

    def student_chain(spec):
        base_example = code_dir / spec["base_example"]
        base_design = code_dir / spec["base_design"]
        example_out = images_dir / students_spec["output_example"].format(cast_id=spec["cast_id"])
        design_out = images_dir / students_spec["output_design"].format(cast_id=spec["cast_id"])
        prompt = students_spec["example_prompt"].format(age=age)
        image_bytes = call_image_api(openrouter_key, prompt, input_references=_image_refs([base_example]), label=f"anchor {spec['cast_id']} example")
        save_jpg(Image.open(io.BytesIO(image_bytes)), example_out)
        prompt = students_spec["design_prompt"].format(age=age)
        image_bytes = call_image_api(openrouter_key, prompt, input_references=_image_refs([example_out, base_design]), label=f"anchor {spec['cast_id']} design")
        save_jpg(Image.open(io.BytesIO(image_bytes)), design_out)
        return {
            "cast_id": spec["cast_id"],
            "is_anchor": True,
            "name": spec["name"],
            "kind": spec["kind"],
            "reference_images": [rel_to_code(example_out, code_dir), rel_to_code(design_out, code_dir)],
            "description": f"{spec['description']}（约 {age} 岁）",
            "presence": spec["presence"],
            "inferred_age": age,
        }

    def classroom_chain():
        # Classroom: base design -> per-project design sheet (same recipe shape as
        # teacher/students; the base image is the identity reference).
        classroom_spec = load_json(specs_dir / "classroom.json")
        regen = classroom_spec["regen"]
        base_design = code_dir / regen["base_design"]
        design_out = images_dir / regen["output_design"]
        prompt = f"{art_style}\n" + regen["prompt"].format(grade_level=grade_level)
        image_bytes = call_image_api(openrouter_key, prompt, input_references=_image_refs([base_design]), label="anchor classroom design")
        save_jpg(Image.open(io.BytesIO(image_bytes)), design_out)
        return {
            "cast_id": classroom_spec["cast_id"],
            "is_anchor": True,
            "name": classroom_spec["name"],
            "kind": classroom_spec["kind"],
            "reference_images": [rel_to_code(design_out, code_dir)],
            "description": classroom_spec["description"],
        }

    chains = [teacher_chain] + [lambda spec=spec: student_chain(spec) for spec in students_spec["students"]] + [classroom_chain]
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(chains))) as pool:
        return list(pool.map(lambda chain: chain(), chains))


def load_art_style(code_dir):
    path = code_dir / "tools" / "prompts" / "art_style.json"
    if not path.exists():
        return ""
    cfg = load_json(path)
    return cfg.get("image", {}).get("styles", "")


def main():
    parser = argparse.ArgumentParser(description="Build cast_set.json from preproduction script/context")
    parser.add_argument("--project-id", required=True, help="Project folder under code/projects/<project_id>")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Kimi model (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    code_dir = Path(__file__).parent.parent
    project_dir = code_dir / "projects" / args.project_id
    casts_dir = project_dir / "casts"
    images_dir = casts_dir / "images"
    casts_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    set_trace_dir(project_dir / "logs")
    load_dotenv(code_dir / ".env")

    script_path = project_dir / "script_final.json"
    context_path = project_dir / "context_pack.json"
    if not script_path.exists():
        print(f"Error: script_final.json not found at {script_path}", file=sys.stderr)
        sys.exit(1)
    if not context_path.exists():
        print(f"Warning: context_pack.json not found at {context_path}; continuing with script only", file=sys.stderr)

    script_obj = load_json(script_path)
    script_text = flatten_script(script_obj)
    chapters = script_obj.get("chapters", []) or []
    chapter_ids = [ch.get("chapter_id") for ch in chapters if isinstance(ch, dict) and ch.get("chapter_id") is not None]
    chapter_list = build_chapter_list(chapters)

    prompt = load_json(code_dir / "tools" / "prompts" / "2_casts.json")
    kimi_key = os.environ.get("OPENROUTER_API_KEY", "")
    kimi_base = os.environ.get("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
    if not kimi_key:
        print("Error: OPENROUTER_API_KEY not set in .env", file=sys.stderr)
        sys.exit(1)
    client = OpenAI(api_key=kimi_key, base_url=kimi_base, timeout=600)

    # Phase 0: regenerate teacher/students/classroom anchors for this project.
    print("\n=== cast phase 0: regenerate anchors ===", file=sys.stderr)
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
    art_style = load_art_style(code_dir)
    anchors = regenerate_anchors(code_dir, project_dir, art_style, openrouter_key)
    print(f"Anchors regenerated: {[a['cast_id'] for a in anchors]}", file=sys.stderr)

    # Phase 1: extract roster
    extract_cfg = prompt["extract"]
    extract_task = (
        extract_cfg["task"]
        .replace("{chapter_list}", chapter_list)
        .replace("{script}", script_text)
    )
    extract_messages = [
        {
            "role": "system",
            "content": extract_cfg["output_format"],
        },
        {"role": "user", "content": extract_task},
    ]
    print("\n=== cast phase 1: extract roster ===", file=sys.stderr)
    roster_obj = call_llm_json(client, extract_messages, args.model, validate_roster_factory(chapter_ids), "extract")
    roster = []
    for item in roster_obj.get("roster", []):
        roster.append(
            {
                "cast_id": item["cast_id"],
                "is_anchor": False,
                "name": item["name"],
                "kind": item["kind"],
                "appears_in_chapters": bucket_to_chapters(item.get("bucket", "consistent"), chapter_ids),
                "description": "",
                "real_image_found": False,
                "source": None,
                "reference_images": [],
            }
        )
    write_json(casts_dir / "roster.json", {"roster": roster})
    print(f"Roster: {len(roster)} non-anchor entities", file=sys.stderr)

    # Phase 2: Wikipedia reference images (parallel; requests are I/O bound)
    print("\n=== cast phase 2: wikipedia reference images ===", file=sys.stderr)
    wiki_manifest = {}

    def fetch_one(entry):
        out_path = images_dir / f"wiki_{entry['cast_id']}.jpg"
        try:
            result = fetch_wiki_image(entry["name"], out_path)
            return entry, result, None
        except Exception as e:  # noqa: BLE001
            result = {"real_image_found": False, "source": None, "wiki_image": None}
            return entry, result, e

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(roster) or 1)) as pool:
        fetch_results = list(pool.map(fetch_one, roster))

    for entry, result, error in fetch_results:
        entry["real_image_found"] = result["real_image_found"]
        entry["source"] = result["source"]
        wiki_manifest[entry["cast_id"]] = result
        if error is not None:
            print(f"  [{entry['cast_id']}] {entry['name']} — ERROR: {error}", file=sys.stderr)
        else:
            status = "OK" if result["real_image_found"] else "MISS"
            print(f"  [{entry['cast_id']}] {entry['name']} — {status}", file=sys.stderr)
    write_json(casts_dir / "wiki_manifest.json", wiki_manifest)
    write_json(casts_dir / "roster.json", {"roster": roster})

    # Phase 3: describe
    descriptions = {}
    if roster:
        describe_cfg = prompt["describe"]
        roster_text = json.dumps({"roster": roster}, ensure_ascii=False, indent=2)
        describe_task = (
            describe_cfg["task"]
            .replace("{roster}", roster_text)
            .replace("{script}", "（见下方消息）")
        )
        content = [{"type": "text", "text": describe_task}]
        missing = []
        for entry in roster:
            info = wiki_manifest.get(entry["cast_id"], {})
            if info.get("real_image_found") and info.get("wiki_image"):
                img_path = images_dir / info["wiki_image"]
                if img_path.exists():
                    content.append({"type": "text", "text": f"--- {entry['cast_id']}（{entry['name']}）---"})
                    content.append({"type": "image_url", "image_url": {"url": image_to_data_url(img_path)}})
                    continue
            missing.append(f"{entry['cast_id']}（{entry['name']}）")
        if missing:
            content.append({"type": "text", "text": f"以下实体无真实参考图片，需凭知识描述并标记 real_image_found=false：{'、'.join(missing)}"})
        content.append({"type": "text", "text": f"以下是完整脚本（供理解语境）：\n\n{script_text}"})

        describe_messages = [
            {
                "role": "system",
                "content": describe_cfg["output_format"],
            },
            {"role": "user", "content": content},
        ]
        print("\n=== cast phase 3: describe ===", file=sys.stderr)
        desc_obj = call_llm_json(client, describe_messages, args.model, validate_descriptions_factory(roster), "describe")
        descriptions = {item["cast_id"]: item["description"] for item in desc_obj.get("casts", [])}
        for entry in roster:
            entry["description"] = descriptions.get(entry["cast_id"], "")
        write_json(casts_dir / "descriptions.json", {"casts": desc_obj.get("casts", [])})
        write_json(casts_dir / "roster.json", {"roster": roster})

    # Phase 4: generate frozen reference sheets (parallel; failures abort as before)
    print("\n=== cast phase 4: generate reference sheets ===", file=sys.stderr)
    if roster and not openrouter_key:
        print("Error: OPENROUTER_API_KEY not set in .env", file=sys.stderr)
        sys.exit(1)

    def generate_one(entry):
        cast_id = entry["cast_id"]
        if entry["kind"] == "character":
            shot = "生成一张统一风格的角色参考图：清晰展示外貌、年龄、体格、服饰、时代背景和标志性细节。"
        else:
            shot = "生成一张统一风格的地点/场景参考图：清晰展示结构、材质、风格、年代、标志性特征和周围环境。"
        gen_prompt = f"{art_style}\n{shot}\n视觉描述：{entry['description']}"
        refs = []
        info = wiki_manifest.get(cast_id, {})
        if info.get("real_image_found") and info.get("wiki_image"):
            wiki_path = images_dir / info["wiki_image"]
            if wiki_path.exists():
                refs.append({"type": "image_url", "image_url": {"url": image_to_data_url(wiki_path)}})
        print(f"  [{cast_id}] generating...", file=sys.stderr)
        image_bytes = call_image_api(openrouter_key, gen_prompt, input_references=refs or None, label=f"cast sheet {cast_id}")
        img = Image.open(io.BytesIO(image_bytes))
        out_path = images_dir / f"{entry['kind']}_{cast_id}_generated.jpg"
        save_jpg(img, out_path)
        return entry, out_path

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(roster) or 1)) as pool:
        for entry, out_path in pool.map(generate_one, roster):
            entry["reference_images"] = [rel_to_code(out_path, code_dir)]
            print(f"  [{entry['cast_id']}] saved {out_path.name}", file=sys.stderr)

    write_json(casts_dir / "roster.json", {"roster": roster})

    cast_set = {
        "project_id": args.project_id,
        "cast": anchors + roster,
    }
    errors = validate_cast_set(cast_set, code_dir)
    if errors:
        write_json(casts_dir / "cast_set.invalid.json", cast_set)
        print("cast_set validation failed:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        sys.exit(2)

    write_json(casts_dir / "cast_set.json", cast_set)
    print(f"\nSaved {casts_dir / 'cast_set.json'} ({len(cast_set['cast'])} cast entries)", file=sys.stderr)


if __name__ == "__main__":
    main()
