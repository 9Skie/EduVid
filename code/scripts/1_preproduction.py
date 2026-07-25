#!/usr/bin/env python3
"""
Steps 1-4 preproduction: one Kimi context, Python-enforced pipeline.

Kimi holds the reasoning context, but this runner owns the order. The model
cannot skip phases; each turn must match the current phase or it is rejected.

Pipeline:
  write_file reading.json
  -> research loop: wiki_search/wiki_fetch, write_file research.json, repeat; finish_research
  -> write_file context_pack.json
  -> write_file script_draft.json
  -> polish_loop: read_file/write_file/count_chinese script_draft.json as needed
  -> finish_script
  -> runner writes projects/<project_id>/context_pack.json and script_final.json
  -> runner synthesizes ElevenLabs audio + timestamp alignments under projects/<project_id>/audio/

Source rule: Wikipedia API only. The model can request wiki_search/wiki_fetch
by query/title/pageid; this script builds zh.wikipedia.org/w/api.php URLs and
runs curl itself. No Baidu, no generic web fetch, no shell.

Usage (run from code/ with venv activated):
    source .venv/bin/activate
    python scripts/1_preproduction.py --project-id triangles
    python scripts/1_preproduction.py --project-id triangles --debug
"""

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import threading
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from elevenlabs import ElevenLabs
from openai import OpenAI


RATE_PER_MIN = 270
DEFAULT_MODEL = os.environ.get("KIMI_MODEL", "kimi-k3")
MAX_TURNS = 80
MAX_PHASE_FAILURES = 6
MAX_WEB_CALLS = 16
MAX_WORKERS = 8  # concurrent per-chapter ElevenLabs TTS calls
FETCH_CHARS = 14000
READ_CHARS = 30000
WIKI_API = "https://zh.wikipedia.org/w/api.php"
WIKI_HOST = "zh.wikipedia.org"
USER_AGENT = "EduVidPreproduction/1.0 (local educational video pipeline)"
ELEVENLABS_VOICES = {
    "female": {"voice_id": os.environ.get("ELEVENLABS_VOICE_FEMALE", "APSIkVZudNbPAwyPoeVO"), "name": "Female narrator"},
    "male": {"voice_id": os.environ.get("ELEVENLABS_VOICE_MALE", "W8lBaQb9YIoddhxfQNLP"), "name": "Male narrator"},
}
ELEVENLABS_VOICE_SETTINGS = {
    "stability": 0.5,
    "similarity_boost": 0.75,
    "style": 0.3,
    "use_speaker_boost": True,
    "speed": 1.2,
}
ELEVENLABS_MODEL_ID = os.environ.get("ELEVENLABS_MODEL_ID", "eleven_v3")
ELEVENLABS_TIMEOUT = 600

PHASES = ["reading", "research", "context", "draft", "polish_loop", "done"]

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


def duration_to_word_limits(target_minutes):
    target = round(target_minutes * RATE_PER_MIN)
    return round(target * 0.90), round(target * 1.10)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def read_text(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def slugify(text, fallback="project"):
    slug = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "-", text).strip("-").lower()
    slug = re.sub(r"-{2,}", "-", slug)
    return slug[:64] or fallback


def chinese_char_count(text):
    """Length rule: Chinese characters only; punctuation/whitespace/ASCII/digits excluded."""
    return len(re.findall(r"[\u4e00-\u9fff]", text or ""))


def script_text(script_obj):
    if not isinstance(script_obj, dict):
        return ""
    parts = []
    for ch in script_obj.get("chapters", []) or []:
        if not isinstance(ch, dict):
            continue
        for line in ch.get("lines", []) or []:
            if isinstance(line, dict) and isinstance(line.get("text"), str):
                parts.append(line["text"])
    return "".join(parts)


def pdf_to_base64_images(pdf_path, page_start, page_end, dpi=100):
    import fitz

    doc = fitz.open(pdf_path)
    total = len(doc)
    print(f"PDF: {total} pages, converting {page_start}-{page_end} at {dpi} DPI", file=sys.stderr)

    images = []
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    for i in range(page_start, page_end + 1):
        if i > total:
            print(f"Page {i} exceeds total ({total}), stopping", file=sys.stderr)
            break
        pix = doc[i - 1].get_pixmap(matrix=mat)
        png = pix.tobytes("png")
        images.append("data:image/png;base64," + base64.b64encode(png).decode("utf-8"))
        print(f"  Page {i}: {len(png) // 1024}KB", file=sys.stderr)
    doc.close()
    return images


def run_curl(url):
    cmd = [
        "curl", "-L",
        "--max-time", "25",
        "--connect-timeout", "10",
        "--compressed",
        "-A", USER_AGENT,
        url,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=35, check=False)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"curl failed: {e}"}
    return {
        "ok": proc.returncode == 0,
        "curl_returncode": proc.returncode,
        "stderr_tail": proc.stderr[-300:],
        "stdout": proc.stdout,
    }


def wiki_search(query, limit=5):
    limit = max(1, min(int(limit or 5), 8))
    params = urllib.parse.urlencode({
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": limit,
        "srprop": "snippet",
    })
    result = run_curl(f"{WIKI_API}?{params}")
    if not result.get("ok"):
        return {"ok": False, "error": result.get("error", "wikipedia search curl failed")}
    try:
        data = json.loads(result["stdout"])
    except json.JSONDecodeError as e:
        return {"ok": False, "error": f"wikipedia search returned non-JSON: {e}"}
    return {
        "ok": True,
        "source": WIKI_API,
        "query": query,
        "hits": [
            {
                "title": item.get("title"),
                "pageid": item.get("pageid"),
                "snippet": re.sub(r"<[^>]+>", "", item.get("snippet", "")),
            }
            for item in data.get("query", {}).get("search", [])
        ],
    }


def wiki_fetch(title=None, pageid=None):
    params = {"action": "query", "prop": "extracts", "explaintext": "1", "redirects": "1", "format": "json"}
    if pageid is not None:
        params["pageids"] = str(pageid)
    elif title:
        params["titles"] = title
    else:
        return {"ok": False, "error": "wiki_fetch needs title or pageid"}

    result = run_curl(f"{WIKI_API}?{urllib.parse.urlencode(params)}")
    if not result.get("ok"):
        return {"ok": False, "error": result.get("error", "wikipedia fetch curl failed")}
    try:
        data = json.loads(result["stdout"])
    except json.JSONDecodeError as e:
        return {"ok": False, "error": f"wikipedia fetch returned non-JSON: {e}"}

    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return {"ok": False, "error": "wikipedia fetch returned no pages"}
    page = next(iter(pages.values()))
    extract = page.get("extract", "") or ""
    pageid = page.get("pageid")
    return {
        "ok": bool(extract),
        "source": WIKI_API,
        "title": page.get("title"),
        "pageid": pageid,
        "url": f"https://zh.wikipedia.org/?curid={pageid}" if pageid else None,
        "text": extract[:FETCH_CHARS],
        "truncated": len(extract) > FETCH_CHARS,
    }


def safe_work_path(work_dir, rel_path):
    if not isinstance(rel_path, str) or not rel_path.strip():
        raise ValueError("path must be a non-empty relative string")
    p = Path(rel_path)
    if p.is_absolute() or any(part == ".." for part in p.parts):
        raise ValueError("workspace path must be relative and contain no ..")
    root = work_dir.resolve()
    candidate = (root / p).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("path escapes workspace")
    return candidate


def tool_write_file(work_dir, rel_path, content):
    try:
        path = safe_work_path(work_dir, rel_path)
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False, indent=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return {"ok": True, "path": str(path.relative_to(work_dir)), "chars": len(content)}


def tool_read_file(work_dir, rel_path):
    try:
        path = safe_work_path(work_dir, rel_path)
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    if not path.exists():
        return {"ok": False, "error": f"file not found: {rel_path}"}
    text = read_text(path)
    return {"ok": True, "path": str(path.relative_to(work_dir)), "text": text[:READ_CHARS], "truncated": len(text) > READ_CHARS}


def tool_count_chinese(work_dir, rel_path, word_min=None, word_max=None):
    try:
        path = safe_work_path(work_dir, rel_path)
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    if not path.exists():
        return {"ok": False, "error": f"file not found: {rel_path}"}

    raw = read_text(path)
    counted_text = raw
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and isinstance(parsed.get("chapters"), list):
            counted_text = script_text(parsed)
    except json.JSONDecodeError:
        pass

    count = chinese_char_count(counted_text)
    result = {
        "ok": True,
        "path": str(path.relative_to(work_dir)),
        "chinese_chars": count,
        "counting_rule": "Chinese characters only; punctuation, whitespace, ASCII letters and digits excluded",
    }
    if word_min is not None and word_max is not None:
        result.update({"word_count_min": word_min, "word_count_max": word_max, "in_range": word_min <= count <= word_max})
    return result


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


def build_initial_messages(pre_prompt, image_urls, project_id, word_min, word_max):
    job = pre_prompt["teacher_job"]
    tb = job["textbook"]
    reading_rules = "\n".join(f"- {r}" for r in pre_prompt.get("reading_rules", []))

    system = (
        "你是 EduVid steps 1-4 的预生产代理。Python runner 强制执行阶段顺序；"
        "不符合当前 phase 的动作会被拒绝。\n\n"
        f"任务：{job.get('task', '')}\n\n"
        f"教学目标：\n{job.get('teaching_objective', '')}\n\n"
        f"学生：{job.get('audience', '')}；grade_level={job.get('grade_level', '')}\n"
        f"教材：《{tb['title']}》第 {tb['page_start']}-{tb['page_end']} 页。\n"
        f"project_id: {project_id}\n"
        f"目标视频长度：{job.get('target_minutes', 14)} 分钟；"
        f"最终脚本中文字数硬范围：{word_min}-{word_max}（只数汉字）。\n\n"
        "阅读规则：\n"
        f"{reading_rules}\n\n"
        "完整预生产提示词 JSON：\n"
        f"{json.dumps(pre_prompt, ensure_ascii=False, indent=2)}\n\n"
        "当前唯一允许的第一动作：write_file reading.json。"
    )

    content = [
        {
            "type": "text",
            "text": (
                f"以下是《{tb['title']}》第 {tb['page_start']}-{tb['page_end']} 页教材页面。"
                "第一阶段只做 reading.json：reading_conclusions + questions。现在禁止 wiki_search/wiki_fetch。"
            ),
        }
    ]
    for url in image_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})
    content.append(
        {
            "type": "text",
            "text": "输出动作：{\"action\":\"write_file\",\"path\":\"reading.json\",\"content\":\"{...}\"}",
        }
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def ids_in_reading(reading):
    point_ids, question_ids = set(), set()
    if isinstance(reading, dict):
        for item in reading.get("reading_conclusions", []) or []:
            if isinstance(item, dict) and item.get("point_id"):
                point_ids.add(item["point_id"])
        for item in reading.get("questions", []) or []:
            if isinstance(item, dict) and item.get("question_id"):
                question_ids.add(item["question_id"])
    return point_ids, question_ids


def validate_reading(reading):
    errors = []
    if not isinstance(reading, dict):
        return ["reading.json must be a JSON object"]
    if not isinstance(reading.get("reading_conclusions"), list) or not reading["reading_conclusions"]:
        errors.append("reading.reading_conclusions must be a non-empty list")
    if not isinstance(reading.get("questions"), list):
        errors.append("reading.questions must be a list")
    point_ids, question_ids = ids_in_reading(reading)
    if len(point_ids) != len(reading.get("reading_conclusions", []) or []):
        errors.append("reading_conclusions point_id must be present and unique")
    if len(question_ids) != len(reading.get("questions", []) or []):
        errors.append("questions question_id must be present and unique")
    return errors


def source_url_ok(url):
    try:
        return urllib.parse.urlparse(url).netloc.lower() == WIKI_HOST
    except Exception:  # noqa: BLE001
        return False


def validate_confirmed(confirmed, reading):
    errors = []
    if not isinstance(confirmed, dict) or not isinstance(confirmed.get("confirmed_answers"), list):
        return ["confirmed_answers.json must be an object with confirmed_answers list"]
    _, question_ids = ids_in_reading(reading)
    seen = set()

    for item in confirmed["confirmed_answers"]:
        if not isinstance(item, dict):
            errors.append("confirmed_answers items must be objects")
            continue
        qid = item.get("question_id")
        status = item.get("status")
        if qid not in question_ids:
            errors.append(f"confirmed answer references unknown question_id {qid}")
        if qid in seen:
            errors.append(f"duplicate confirmed answer for {qid}")
        seen.add(qid)
        if status not in ("confirmed", "not_confirmed"):
            errors.append(f"{qid} status must be confirmed or not_confirmed")
            continue

        source = item.get("source")
        if isinstance(source, dict) and "type" in source:
            errors.append(f"{qid} source.type is not allowed; Wikipedia is implied")
        if status == "confirmed":
            if not item.get("answer"):
                errors.append(f"{qid} confirmed answer needs answer text")
            if not isinstance(source, dict):
                errors.append(f"{qid} confirmed answer needs source object")
            else:
                if not source.get("title") and not source.get("pageid"):
                    errors.append(f"{qid} source needs title or pageid")
                if source.get("url") and not source_url_ok(source["url"]):
                    errors.append(f"{qid} source.url must be zh.wikipedia.org")
        elif source not in (None, {}):
            errors.append(f"{qid} not_confirmed source must be null")

    missing = question_ids - seen
    if missing:
        errors.append(f"questions missing confirmed/not_confirmed answers: {sorted(missing)}")
    return errors


def validate_research_work(research, allow_open):
    errors = []
    if not isinstance(research, dict):
        return ["research.json must be a JSON object"]
    questions = research.get("questions")
    answers = research.get("confirmed_answers")
    if not isinstance(questions, list):
        errors.append("research.questions must be a list")
        questions = []
    if not isinstance(answers, list):
        errors.append("research.confirmed_answers must be a list")
        answers = []

    q_status = {}
    for q in questions:
        if not isinstance(q, dict):
            errors.append("research.questions items must be objects")
            continue
        qid = q.get("question_id")
        status = q.get("status", "open")
        if not qid:
            errors.append("research question missing question_id")
            continue
        if qid in q_status:
            errors.append(f"duplicate research question_id {qid}")
        if status not in ("open", "confirmed", "not_confirmed"):
            errors.append(f"{qid} status must be open, confirmed, or not_confirmed")
        q_status[qid] = status

    answered = set()
    for item in answers:
        if not isinstance(item, dict):
            errors.append("research.confirmed_answers items must be objects")
            continue
        qid = item.get("question_id")
        status = item.get("status")
        if qid not in q_status:
            errors.append(f"confirmed answer references unknown research question_id {qid}")
            continue
        answered.add(qid)
        if status not in ("confirmed", "not_confirmed"):
            errors.append(f"{qid} answer status must be confirmed or not_confirmed")
        if q_status.get(qid) != "open" and status != q_status.get(qid):
            errors.append(f"{qid} question status {q_status.get(qid)} does not match answer status {status}")

        source = item.get("source")
        if isinstance(source, dict) and "type" in source:
            errors.append(f"{qid} source.type is not allowed; Wikipedia is implied")
        if status == "confirmed":
            if not item.get("answer"):
                errors.append(f"{qid} confirmed answer needs answer text")
            if not isinstance(source, dict):
                errors.append(f"{qid} confirmed answer needs source object")
            else:
                if not source.get("title") and not source.get("pageid"):
                    errors.append(f"{qid} source needs title or pageid")
                if source.get("url") and not source_url_ok(source["url"]):
                    errors.append(f"{qid} source.url must be zh.wikipedia.org")
        elif status == "not_confirmed" and source not in (None, {}):
            errors.append(f"{qid} not_confirmed source must be null")

    for qid, status in q_status.items():
        if status in ("confirmed", "not_confirmed") and qid not in answered:
            errors.append(f"{qid} is marked {status} but has no confirmed_answers item")
        if not allow_open and status == "open":
            errors.append(f"{qid} is still open; finish_research requires all questions confirmed or not_confirmed")
    return errors


def context_ids(context_pack):
    point_ids, question_ids = set(), set()
    if isinstance(context_pack, dict):
        for key in ("reading_conclusions", "teaching_points"):
            for item in context_pack.get(key, []) or []:
                if isinstance(item, dict) and item.get("point_id"):
                    point_ids.add(item["point_id"])
        for key in ("questions", "confirmed_answers"):
            for item in context_pack.get(key, []) or []:
                if isinstance(item, dict) and item.get("question_id"):
                    question_ids.add(item["question_id"])
    return point_ids, question_ids


def validate_context_pack(context_pack):
    errors = []
    if not isinstance(context_pack, dict):
        return ["context_pack must be a JSON object"]
    for key in ("reading_conclusions", "questions", "confirmed_answers", "teaching_points"):
        if not isinstance(context_pack.get(key), list):
            errors.append(f"context_pack.{key} must be a list")
    errors.extend(validate_reading(context_pack))
    errors.extend(validate_confirmed(context_pack, context_pack))
    for q in context_pack.get("questions", []) or []:
        if isinstance(q, dict) and q.get("status") == "open":
            errors.append(f"context_pack question {q.get('question_id')} is still open; finish research first")

    point_ids, question_ids = context_ids(context_pack)
    for item in context_pack.get("teaching_points", []) or []:
        if not isinstance(item, dict):
            errors.append("teaching_points items must be objects")
            continue
        if not item.get("point_id") or not item.get("point"):
            errors.append("teaching_point needs point_id and point")
        for ref in item.get("based_on", []) or []:
            if ref not in point_ids and ref not in question_ids:
                errors.append(f"teaching_point {item.get('point_id')} based_on unknown ref {ref}")
    return errors


def validate_script(script, context_pack, word_min, word_max, check_length=False, check_style=False):
    errors = []
    if not isinstance(script, dict) or not isinstance(script.get("chapters"), list) or not script["chapters"]:
        return ["script must be an object with non-empty chapters list"]

    point_ids, question_ids = context_ids(context_pack)
    valid_refs = point_ids | question_ids
    seen_chapters, seen_lines = set(), set()

    for ch in script["chapters"]:
        if not isinstance(ch, dict):
            errors.append("each chapter must be an object")
            continue
        cid = ch.get("chapter_id")
        if cid in seen_chapters:
            errors.append(f"duplicate chapter_id {cid}")
        seen_chapters.add(cid)
        if not ch.get("title"):
            errors.append(f"chapter {cid} missing title")
        lines = ch.get("lines")
        if not isinstance(lines, list) or not lines:
            errors.append(f"chapter {cid} needs non-empty lines")
            continue
        for line in lines:
            if not isinstance(line, dict):
                errors.append("each line must be an object")
                continue
            line_id = line.get("line_id")
            if not line_id:
                errors.append("line missing line_id")
            elif line_id in seen_lines:
                errors.append(f"duplicate line_id {line_id}")
            else:
                seen_lines.add(line_id)
            if not isinstance(line.get("text"), str) or not line["text"].strip():
                errors.append(f"{line_id or 'line'} missing text")
            for ref in line.get("point_ids", []) or []:
                if ref not in valid_refs:
                    errors.append(f"{line_id} references unknown point/question id {ref}")

    text = script_text(script)
    if check_length:
        count = chinese_char_count(text)
        if count < word_min or count > word_max:
            errors.append(f"script chinese char count {count} outside required range {word_min}-{word_max}")
    if check_style and ("我" in text or "我们" in text):
        errors.append("script contains first-person 我/我们, which script voice rules forbid")
    return errors


def phase_requirement(phase):
    return {
        "reading": "write_file reading.json",
        "research": "wiki_search/wiki_fetch, write_file research.json with open/closed questions, repeat; finish_research when no question is open",
        "context": "write_file context_pack.json",
        "draft": "write_file script_draft.json rough draft",
        "polish_loop": "read_file/write_file/count_chinese script_draft.json as needed, then finish_script",
        "done": "done",
    }[phase]


def load_work_json(work_dir, rel_path):
    return json.loads(read_text(safe_work_path(work_dir, rel_path)))


def run_file_action(action_obj, work_dir, word_min, word_max):
    action = action_obj.get("action")
    path = action_obj.get("path", "")
    if action == "write_file":
        return tool_write_file(work_dir, path, action_obj.get("content", ""))
    if action == "read_file":
        return tool_read_file(work_dir, path)
    if action == "count_chinese":
        return tool_count_chinese(work_dir, path, word_min, word_max)
    return {"ok": False, "error": f"unsupported file action {action}"}


def execute_phase_action(action_obj, work_dir, state, word_min, word_max):
    phase = state["phase"]
    action = action_obj.get("action")
    path = action_obj.get("path", "")

    if phase == "research":
        if action in ("wiki_search", "wiki_fetch"):
            if state["web_calls"] >= MAX_WEB_CALLS:
                return {"ok": False, "error": f"web action budget exhausted ({MAX_WEB_CALLS})"}, False, ["web budget exhausted"]
            state["web_calls"] += 1
            result = wiki_search(action_obj.get("query", ""), action_obj.get("limit", 5)) if action == "wiki_search" else wiki_fetch(
                title=action_obj.get("title"), pageid=action_obj.get("pageid")
            )
            return result, False, [] if result.get("ok") else [result.get("error", "wiki action failed")]

        if action == "write_file" and path == "research.json":
            result = tool_write_file(work_dir, path, action_obj.get("content", ""))
            if not result.get("ok"):
                return result, False, [result.get("error", "tool failed")]
            try:
                errors = validate_research_work(load_work_json(work_dir, "research.json"), allow_open=True)
            except Exception as e:  # noqa: BLE001
                errors = [f"research write validation crashed: {e}"]
            return result, False, errors

        if action == "finish_research":
            try:
                errors = validate_research_work(load_work_json(work_dir, "research.json"), allow_open=False)
            except Exception as e:  # noqa: BLE001
                errors = [f"finish_research validation crashed: {e}"]
            return {"ok": not errors, "action": "finish_research", "errors": errors}, not errors, errors

        error = (
            "PHASE_ERROR: current phase is research; allowed actions are wiki_search/wiki_fetch, "
            "write_file research.json, or finish_research; "
            f"got {action} {path}"
        )
        return {"ok": False, "error": error, "required_action": phase_requirement(phase)}, False, [error]

    if phase == "polish_loop":
        if action == "finish_script":
            try:
                errors = validate_script(
                    load_work_json(work_dir, "script_draft.json"),
                    load_work_json(work_dir, "context_pack.json"),
                    word_min,
                    word_max,
                    check_length=True,
                    check_style=True,
                )
            except Exception as e:  # noqa: BLE001
                errors = [f"finish_script validation crashed: {e}"]
            return {"ok": not errors, "action": "finish_script", "errors": errors}, not errors, errors

        if action not in ("read_file", "write_file", "count_chinese") or path != "script_draft.json":
            error = (
                "PHASE_ERROR: current phase is polish_loop; allowed actions are "
                "read_file/write_file/count_chinese on script_draft.json, or finish_script; "
                f"got {action} {path}"
            )
            return {"ok": False, "error": error, "required_action": phase_requirement(phase)}, False, [error]

        result = run_file_action(action_obj, work_dir, word_min, word_max)
        if not result.get("ok"):
            return result, False, [result.get("error", "tool failed")]

        errors = []
        if action == "write_file":
            try:
                errors = validate_script(
                    load_work_json(work_dir, "script_draft.json"),
                    load_work_json(work_dir, "context_pack.json"),
                    word_min,
                    word_max,
                    check_length=False,
                    check_style=False,
                )
            except Exception as e:  # noqa: BLE001
                errors = [f"polish write validation crashed: {e}"]
        return result, False, errors

    required = {
        "reading": ("write_file", "reading.json"),
        "context": ("write_file", "context_pack.json"),
        "draft": ("write_file", "script_draft.json"),
    }.get(phase)

    if not required:
        return {"ok": False, "error": f"phase {phase} accepts no further actions"}, False, [f"phase {phase} accepts no further actions"]

    req_action, req_path = required
    if action != req_action or path != req_path:
        error = f"PHASE_ERROR: current phase is {phase}; required action is {req_action} {req_path}; got {action} {path}"
        return {"ok": False, "error": error, "required_action": phase_requirement(phase)}, False, [error]

    result = run_file_action(action_obj, work_dir, word_min, word_max)
    if not result.get("ok"):
        return result, False, [result.get("error", "tool failed")]

    errors = []
    advance = False
    try:
        if phase == "reading":
            errors = validate_reading(load_work_json(work_dir, "reading.json"))
            advance = not errors
        elif phase == "context":
            errors = validate_context_pack(load_work_json(work_dir, "context_pack.json"))
            advance = not errors
        elif phase == "draft":
            errors = validate_script(
                load_work_json(work_dir, "script_draft.json"),
                load_work_json(work_dir, "context_pack.json"),
                word_min,
                word_max,
                check_length=False,
                check_style=False,
            )
            advance = not errors
    except Exception as e:  # noqa: BLE001
        errors = [f"phase {phase} validation crashed: {e}"]
        advance = False

    return result, advance, errors


def finalize(project_dir, work_dir, project_id, model, word_min, word_max, state, tool_log, debug):
    context_pack = load_work_json(work_dir, "context_pack.json")
    script_final = load_work_json(work_dir, "script_draft.json")
    now = datetime.now(timezone.utc).isoformat()
    count = chinese_char_count(script_text(script_final))

    context_pack["_meta"] = {
        "project_id": project_id,
        "model": model,
        "created_at": now,
        "word_count_min": word_min,
        "word_count_max": word_max,
        "web_calls": state["web_calls"],
        "source": WIKI_API,
    }
    script_final["_meta"] = {
        "project_id": project_id,
        "model": model,
        "created_at": now,
        "word_count_min": word_min,
        "word_count_max": word_max,
        "chinese_chars": count,
        "verification": "rough write -> polish_loop read/write/count_chinese -> finish_script",
    }

    context_out = project_dir / "context_pack.json"
    script_out = project_dir / "script_final.json"
    write_json(context_out, context_pack)
    write_json(script_out, script_final)

    if debug:
        trace_out = project_dir / "preproduction_trace.json"
        write_json(
            trace_out,
            {
                "project_id": project_id,
                "model": model,
                "created_at": now,
                "web_calls": state["web_calls"],
                "tool_log": tool_log,
                "prompt_sections": ["teacher_job", "reading_rules", "script_reference", "script_voice_rules", "script_length_rule"],
            },
        )
        print(f"Saved trace to {trace_out}", file=sys.stderr)

    print(f"\nSaved {context_out}", file=sys.stderr)
    print(f"Saved {script_out}", file=sys.stderr)


def chapter_tts_text(chapter):
    """Narration text for one chapter: script line texts concatenated, no titles/metadata."""
    parts = []
    if not isinstance(chapter, dict):
        return ""
    for line in chapter.get("lines", []) or []:
        if isinstance(line, dict) and isinstance(line.get("text"), str) and line["text"].strip():
            parts.append(line["text"].strip())
    return "".join(parts)


def alignment_to_dict(align):
    if align is None:
        raise RuntimeError("ElevenLabs returned no alignment data")
    return {
        "characters": list(align.characters),
        "char_start_times": list(align.character_start_times_seconds),
        "char_end_times": list(align.character_end_times_seconds),
    }


def compute_script_line_timings(chapter, alignment):
    """Consume ElevenLabs character alignment by script line length, preserving line_id."""
    chars = alignment["characters"]
    starts = alignment["char_start_times"]
    ends = alignment["char_end_times"]
    out = []
    idx = 0

    for line in chapter.get("lines", []) or []:
        if not isinstance(line, dict):
            continue
        text = (line.get("text") or "").strip()
        if not text:
            continue
        n = len(text)
        start = starts[idx] if idx < len(starts) else None
        end_idx = idx + n - 1
        end = ends[end_idx] if end_idx < len(ends) else None
        out.append(
            {
                "line_id": line.get("line_id"),
                "text": text,
                "start": round(start, 3) if start is not None else None,
                "end": round(end, 3) if end is not None else None,
                "start_ms": round(start * 1000) if start is not None else None,
                "end_ms": round(end * 1000) if end is not None else None,
            }
        )
        idx += n

    return out, idx, len(chars)


def synthesize_chapter_audio(client, voice_cfg, audio_dir, voice_tag, chapter, chapter_id, title):
    """Synthesize one chapter's audio + alignment + line timings; returns manifest entry or None."""
    text = chapter_tts_text(chapter)
    if not text:
        print(f"  Chapter {chapter_id}: empty, skipped", file=sys.stderr)
        return None
    if len(text) > 5000:
        raise RuntimeError(f"Chapter {chapter_id} is {len(text)} chars, over ElevenLabs v3 5000-char limit; split the chapter")

    print(f"  Chapter {chapter_id}: {title} ({len(text)} chars)", file=sys.stderr)
    response = client.text_to_speech.convert_with_timestamps(
        voice_id=voice_cfg["voice_id"],
        text=text,
        model_id=ELEVENLABS_MODEL_ID,
        voice_settings=ELEVENLABS_VOICE_SETTINGS,
    )
    audio_bytes = base64.b64decode(response.audio_base_64)
    alignment = alignment_to_dict(response.alignment)
    lines, consumed, total_chars = compute_script_line_timings(chapter, alignment)
    if consumed != total_chars:
        print(f"    warning: consumed {consumed}/{total_chars} alignment chars", file=sys.stderr)

    tag = f"voice_chapter_{chapter_id}_{voice_tag}"
    audio_path = audio_dir / f"{tag}.mp3"
    align_path = audio_dir / f"{tag}_alignment.json"
    lines_path = audio_dir / f"{tag}_lines.json"
    audio_path.write_bytes(audio_bytes)
    write_json(align_path, alignment)
    write_json(lines_path, lines)

    duration_s = alignment["char_end_times"][-1] if alignment["char_end_times"] else 0
    result = {
        "chapter": chapter_id,
        "name": title,
        "voice": voice_cfg["name"],
        "audio": audio_path.name,
        "alignment": align_path.name,
        "lines": lines_path.name,
        "duration_s": round(duration_s, 2),
        "char_count": len(text),
    }

    manifest = {
        f"voice_{voice_tag}": {
            "voice_id": voice_cfg["voice_id"],
            "voice_name": voice_cfg["name"],
            "model": ELEVENLABS_MODEL_ID,
            "settings": ELEVENLABS_VOICE_SETTINGS,
            "chapters": [result],
            "total_duration_s": result["duration_s"],
        }
    }
    write_json(audio_dir / f"manifest_chapter_{chapter_id}.json", manifest)
    print(f"    saved {audio_path.name} + alignment + lines ({duration_s:.1f}s)", file=sys.stderr)
    return result


def synthesize_audio(project_dir, voice_tag):
    """Turn finalized script_final.json into per-chapter ElevenLabs audio + timestamps."""
    script_path = project_dir / "script_final.json"
    if not script_path.exists():
        print(f"Audio skipped: {script_path} not found", file=sys.stderr)
        return

    api_key = os.environ.get("ELEVENLABS_API_KEY", "")
    if not api_key:
        print("Audio skipped: ELEVENLABS_API_KEY not set in .env", file=sys.stderr)
        return

    voice_tag = str(voice_tag or "").strip().lower()
    voice_cfg = ELEVENLABS_VOICES.get(voice_tag)
    if not voice_cfg:
        print(f"Audio skipped: no ElevenLabs voice for narration_gender={voice_tag!r}", file=sys.stderr)
        return

    script_obj = load_json(script_path)
    chapters = script_obj.get("chapters", []) or []
    if not chapters:
        print("Audio skipped: script has no chapters", file=sys.stderr)
        return

    client = ElevenLabs(api_key=api_key, timeout=ELEVENLABS_TIMEOUT)
    audio_dir = project_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== ElevenLabs audio: {ELEVENLABS_MODEL_ID} voice={voice_cfg['name']} ({min(MAX_WORKERS, len(chapters))} parallel) ===", file=sys.stderr)

    def one(args):
        i, chapter = args
        chapter_id = chapter.get("chapter_id", i)
        title = chapter.get("title", f"Chapter {chapter_id}")
        return synthesize_chapter_audio(client, voice_cfg, audio_dir, voice_tag, chapter, chapter_id, title)

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(chapters))) as pool:
        results = [r for r in pool.map(one, list(enumerate(chapters, 1))) if r]

    write_json(
        audio_dir / "audio_manifest.json",
        {
            "project_id": project_dir.name,
            "voice": voice_tag,
            "voice_id": voice_cfg["voice_id"],
            "model": ELEVENLABS_MODEL_ID,
            "settings": ELEVENLABS_VOICE_SETTINGS,
            "chapters": results,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(f"Saved {audio_dir / 'audio_manifest.json'}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Enforced one-context Kimi preproduction for steps 1-4")
    parser.add_argument("--project-id", default=None, help="Output folder under code/projects/<project_id>")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Kimi model (default: {DEFAULT_MODEL})")
    parser.add_argument("--debug", action="store_true", help="Also write preproduction_trace.json")
    parser.add_argument("--skip-tts", action="store_true", help="Do not synthesize ElevenLabs audio/timestamps after finalizing the script")
    parser.add_argument("--voice", choices=["female", "male"], default=None, help="Override narration voice; default is teacher_job.narration_gender")
    args = parser.parse_args()

    code_dir = Path(__file__).parent.parent
    tools_dir = code_dir / "tools"
    load_dotenv(code_dir / ".env")

    api_key = os.environ.get("KIMI_API_KEY", "")
    api_base = os.environ.get("KIMI_API_BASE", "https://api.moonshot.cn/v1")
    if not api_key:
        print("Error: KIMI_API_KEY not set in .env", file=sys.stderr)
        sys.exit(1)

    pre_prompt_path = tools_dir / "prompts" / "1_preproduction.json"
    if not pre_prompt_path.exists():
        print(f"Error: preproduction prompt not found at {pre_prompt_path}", file=sys.stderr)
        sys.exit(1)
    pre_prompt = load_json(pre_prompt_path)
    job = pre_prompt["teacher_job"]
    tb = job["textbook"]

    project_id = args.project_id or slugify(f"{tb.get('title', 'textbook')}-p{tb.get('page_start')}-{tb.get('page_end')}")
    project_dir = code_dir / "projects" / project_id
    work_dir = project_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    set_trace_dir(project_dir / "logs")

    word_min, word_max = duration_to_word_limits(job.get("target_minutes", 14))
    print(f"Project: {project_id} | target {job.get('target_minutes', 14)} min -> {word_min}-{word_max} chinese chars", file=sys.stderr)
    print(f"Workspace: {work_dir}", file=sys.stderr)

    pdf_path = tools_dir / "dataset" / tb["pdf_path"]
    if not pdf_path.exists():
        print(f"Error: PDF not found at {pdf_path}", file=sys.stderr)
        sys.exit(1)

    image_urls = pdf_to_base64_images(str(pdf_path), tb["page_start"], tb["page_end"])
    print(f"\n{len(image_urls)} pages ready, calling {args.model}...", file=sys.stderr)

    client = OpenAI(api_key=api_key, base_url=api_base, timeout=600)
    messages = build_initial_messages(pre_prompt, image_urls, project_id, word_min, word_max)

    state = {"phase": "reading", "web_calls": 0, "failures": 0}
    tool_log = []

    for turn in range(1, MAX_TURNS + 1):
        print(f"\n=== phase {state['phase']} | turn {turn}/{MAX_TURNS} ===", file=sys.stderr)
        raw = stream_json(client, messages, args.model, label=f"phase {state['phase']} turn {turn}")

        try:
            action_obj = json.loads(raw)
        except json.JSONDecodeError as e:
            action_obj = {"action": "__invalid_json__", "error": str(e)}

        messages.append({"role": "assistant", "content": raw})
        old_phase = state["phase"]
        result, advance, errors = execute_phase_action(action_obj, work_dir, state, word_min, word_max)
        tool_log.append({"turn": turn, "phase": old_phase, "action": action_obj.get("action"), "ok": result.get("ok"), "advance": advance})

        if advance:
            state["phase"] = PHASES[PHASES.index(old_phase) + 1]
            state["failures"] = 0
        elif result.get("ok") and not errors:
            state["failures"] = 0
        else:
            state["failures"] += 1
            if state["failures"] > MAX_PHASE_FAILURES:
                print(f"Error: phase {old_phase} failed too many times", file=sys.stderr)
                sys.exit(3)

        if state["phase"] == "done":
            finalize(project_dir, work_dir, project_id, args.model, word_min, word_max, state, tool_log, args.debug)
            if not args.skip_tts:
                synthesize_audio(project_dir, args.voice or job.get("narration_gender"))
            return

        messages.append(
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "tool_result": result,
                        "validation_errors": errors,
                        "phase_control": {
                            "phase": state["phase"],
                            "required_action": phase_requirement(state["phase"]),
                            "failures_in_phase": state["failures"],
                        },
                    },
                    ensure_ascii=False,
                ),
            }
        )

    print(f"Error: preproduction did not finish within {MAX_TURNS} turns", file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
