#!/usr/bin/env python3
"""
Stage 4: Write video script from the analysis blueprint.

Three steps, each operating on a single shared file (tmp/script.json) via
a tool-calling agent loop:
  Draft:      DeepSeek calls write_script to create the initial script.
  Fact-check: Kimi reads the script + textbook pages, calls edit_script
              to fix each factual error, then finish.
  Humanize:   DeepSeek reads the script, calls edit_script to polish
              voice and land the word band, then finish.

Usage (run from code/ with venv activated):
    source .venv/bin/activate
    python scripts/4_research_to_script.py
    python scripts/4_research_to_script.py --only draft
    python scripts/4_research_to_script.py --only fact_check
    python scripts/4_research_to_script.py --only humanize
"""

import argparse
import base64
import json
import os
import re
import shutil
import sys
from pathlib import Path

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


# --- Helpers ---

def load_json_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def count_chinese_chars(text):
    count = 0
    for ch in text:
        code = ord(ch)
        if (0x4E00 <= code <= 0x9FFF or
            0x3400 <= code <= 0x4DBF or
            0xF900 <= code <= 0xFAFF):
            count += 1
    return count


def count_chapter_chars(raw_json):
    """Count total Chinese chars across all chapter scripts in the JSON."""
    try:
        obj = json.loads(raw_json)
        total = 0
        for ch in obj.get("chapters", []):
            total += count_chinese_chars(ch.get("script", ""))
        return total
    except json.JSONDecodeError:
        return count_chinese_chars(raw_json)


_LABEL_RE = re.compile(
    r"^(?:第[一二三四五六七八九十百千零〇\d]+[章节回]"
    r"|chapter\s*\d+|引子|序章|前言|导言|楔子|尾声|结语|开篇"
    r"|prologue|epilogue)"
    r"[\s：:]*",
    re.IGNORECASE,
)


def force_chapter_numbering(script_path):
    """Prepend 'chapter N: ' to each chapter name, stripping any existing label.

    Kills ambiguity from titles like '引子：...' — every chapter becomes
    unambiguously numbered so downstream stages can map entities to chapters.
    """
    raw = script_path.read_text(encoding="utf-8")
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return
    chapters = obj.get("chapters", [])
    if not chapters:
        return
    for i, ch in enumerate(chapters, 1):
        name = _LABEL_RE.sub("", (ch.get("name") or "").strip())
        ch["name"] = f"chapter {i}: {name}"
    script_path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def pdf_to_base64_images(pdf_path, page_start, page_end, dpi=100):
    """Convert PDF pages (1-indexed, inclusive) to base64-encoded PNG strings."""
    import fitz

    doc = fitz.open(pdf_path)
    total = len(doc)
    print(f"PDF: {total} pages, converting {page_start}-{page_end} at {dpi} DPI", file=sys.stderr)

    images = []
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)

    for i in range(page_start, page_end + 1):
        if i > total:
            break
        pix = doc[i - 1].get_pixmap(matrix=mat)
        png = pix.tobytes("png")
        b64 = base64.b64encode(png).decode("utf-8")
        images.append(f"data:image/png;base64,{b64}")
        print(f"  Page {i}: {len(png) // 1024}KB", file=sys.stderr)

    doc.close()
    return images


def load_word_limits(textbook_path):
    """Read word_count_min/max from stage 1's _meta. Defaults to the 14-min band."""
    default_min, default_max = 3402, 4158
    if not textbook_path.exists():
        return default_min, default_max
    try:
        obj = json.loads(load_text(textbook_path))
        meta = obj.get("_meta", {})
        return meta.get("word_count_min", default_min), meta.get("word_count_max", default_max)
    except json.JSONDecodeError:
        return default_min, default_max


def load_teacher_meta(project_dir):
    """Load teaching_objective + textbook page range from the teacher prompt JSON."""
    teacher = load_json_file(project_dir / "prompts" / "1_teacher.json")
    return {
        "teaching_objective": teacher.get("teaching_objective", ""),
        "textbook": teacher.get("textbook", {}),
    }


def _fill(text, word_min, word_max, teaching_objective=""):
    """Substitute prompt placeholders."""
    return (text
            .replace("{word_count_min}", str(word_min))
            .replace("{word_count_max}", str(word_max))
            .replace("{teaching_objective}", teaching_objective)
            .replace("{textbook_pdf_portion}", "（见下方消息中的教材页面图片）"))


def format_analysis_brief(analysis):
    """Turn the stage-3 analysis JSON into a readable text block for the LLM."""
    lines = []
    lines.append("【视频概览】")
    lines.append(analysis.get("summary", ""))
    lines.append("")

    lines.append("【叙事弧线】")
    lines.append(analysis.get("narrative_line", ""))
    lines.append("")

    lines.append("【知识板块】")
    for block in analysis.get("blocks", []):
        lines.append(f"━━━ {block.get('name', '?')} ━━━")
        lines.append(block.get("content", ""))
        if block.get("textbook_points"):
            lines.append(f"教材依据：{block['textbook_points']}")
        if block.get("research_points"):
            lines.append(f"研究补充：{block['research_points']}")
        lines.append("")

    return "\n".join(lines)


def _format_rules(rules):
    """Render the 自然表达准则 dict/list/string into readable text."""
    if isinstance(rules, dict):
        parts = []
        for k, v in rules.items():
            if isinstance(v, list):
                parts.append(f"{k}：")
                for item in v:
                    parts.append(f"  - {item}")
            else:
                parts.append(f"{k}：{v}")
        return "\n".join(parts)
    if isinstance(rules, list):
        return "\n".join(f"- {item}" for item in rules)
    return str(rules)


def build_reference_block(prompt_cfg):
    """Build the reference script block shared by all steps."""
    ref = prompt_cfg["reference"]
    return f"\n\n─── 范例参考 ───\n{ref['guidance']}\n\n{ref['script']}\n\n─── 范例亮点总结 ───\n{ref['summary']}"


def build_client(cfg):
    """Create an OpenAI client from a provider config."""
    api_key = os.environ.get(cfg["api_key_env"], "")
    api_base = os.environ.get(cfg["api_base_env"], cfg["api_base_default"])
    if not api_key:
        print(f"Error: {cfg['api_key_env']} not set in .env", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=api_key, base_url=api_base, timeout=600)


# --- Tool definitions ---

TOOL_WRITE_SCRIPT = {
    "type": "function",
    "function": {
        "name": "write_script",
        "description": (
            "Write the complete script to the script file (overwrites everything). "
            "Use this to create the initial draft. The content must be valid JSON "
            "with the chapter structure."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The complete script as a JSON string.",
                }
            },
            "required": ["content"],
        },
    },
}

TOOL_EDIT_SCRIPT = {
    "type": "function",
    "function": {
        "name": "edit_script",
        "description": (
            "Replace an exact substring in the script file with new text. "
            "old_text must match the raw file content byte-for-byte, including "
            "JSON escaping (quotes, backslashes, whitespace). Only the matched "
            "substring is replaced; everything else stays unchanged. If old_text "
            "appears multiple times, provide a longer old_text to be unique."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "old_text": {
                    "type": "string",
                    "description": "The exact text to find in the current script file.",
                },
                "new_text": {
                    "type": "string",
                    "description": "The replacement text.",
                },
            },
            "required": ["old_text", "new_text"],
        },
    },
}

TOOL_READ_SCRIPT = {
    "type": "function",
    "function": {
        "name": "read_script",
        "description": (
            "Read and return the current content of the script file. "
            "Call this to see the current state before making edits."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

TOOL_FINISH = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": "Signal that you are done. Call this when the script is complete and ready.",
        "parameters": {"type": "object", "properties": {}},
    },
}


def get_tools_for_step(step):
    """Return available tools for a given step."""
    if step == "draft":
        return [TOOL_WRITE_SCRIPT, TOOL_READ_SCRIPT, TOOL_FINISH]
    return [TOOL_EDIT_SCRIPT, TOOL_READ_SCRIPT, TOOL_FINISH]


# --- Tool execution ---

def execute_tool(name, args, script_path):
    """Execute a tool call on the script file. Returns a result string."""
    if name == "write_script":
        content = args.get("content", "")
        script_path.parent.mkdir(exist_ok=True)
        script_path.write_text(content, encoding="utf-8")
        chars = count_chapter_chars(content)
        return f"脚本已写入文件（{chars} 个汉字）。"

    if name == "read_script":
        if not script_path.exists():
            return "错误：脚本文件尚不存在。"
        return script_path.read_text(encoding="utf-8")

    if name == "edit_script":
        old_text = args.get("old_text", "")
        new_text = args.get("new_text", "")
        if not script_path.exists():
            return "错误：脚本文件不存在。请先调用 read_script 查看。"
        if not old_text:
            return "错误：old_text 不能为空。"
        current = script_path.read_text(encoding="utf-8")
        occurrences = current.count(old_text)
        if occurrences == 0:
            return (
                "错误：old_text 在脚本中未找到。请确保从文件中精确复制文本"
                "（包括 JSON 转义符号、引号、空格）。请调用 read_script 查看当前内容后重试。"
            )
        if occurrences > 1:
            return (
                f"错误：old_text 在脚本中匹配到 {occurrences} 处。"
                f"请提供更长、更唯一的 old_text 以精确匹配一处。"
            )
        new_content = current.replace(old_text, new_text)
        script_path.write_text(new_content, encoding="utf-8")
        chars = count_chapter_chars(new_content)
        return f"修改已应用（当前共 {chars} 个汉字）。"

    if name == "finish":
        return "已完成。"

    return f"未知工具：{name}"


# --- Agent loop ---

def call_with_tools(client, cfg, messages, tools):
    """Non-streaming LLM call with tool support."""
    kwargs = dict(
        model=cfg["model"],
        messages=messages,
        max_tokens=cfg["max_tokens"],
        tools=tools,
        stream=False,
    )
    if cfg.get("reasoning_effort"):
        kwargs["reasoning_effort"] = cfg["reasoning_effort"]
    if cfg["extra_body"]:
        kwargs["extra_body"] = cfg["extra_body"]

    return client.chat.completions.create(**kwargs)


def agent_loop(client, cfg, system_prompt, user_content, step, script_path,
               max_iterations=50, word_check=None):
    """Run a tool-calling agent loop until finish() is called or max iterations.

    word_check: optional tuple (word_min, word_max, max_retries) for humanize.
    When set, finish() is rejected if the script is outside the word band;
    feedback is sent and the loop continues.
    """
    tools = get_tools_for_step(step)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    word_min = word_max = word_max_retries = None
    word_retries = 0
    if word_check:
        word_min, word_max, word_max_retries = word_check

    finished = False
    for iteration in range(1, max_iterations + 1):
        response = call_with_tools(client, cfg, messages, tools)
        msg = response.choices[0].message

        assistant_msg = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments or "{}",
                    },
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_msg)

        if not msg.tool_calls:
            messages.append({
                "role": "user",
                "content": "请使用工具完成任务。完成后调用 finish。",
            })
            continue

        finish_in_batch = False
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError:
                args = {}

            print(f"  → {fn_name}", file=sys.stderr)
            result = execute_tool(fn_name, args, script_path)

            if fn_name == "read_script":
                print(f"    ({len(result)} chars returned)", file=sys.stderr)
            else:
                snippet = result[:150].replace("\n", " ")
                print(f"    {snippet}", file=sys.stderr)

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

            if fn_name == "finish":
                finish_in_batch = True

        if not finish_in_batch:
            continue

        # finish was called — check word count if required
        if word_check and script_path.exists():
            chars = count_chapter_chars(script_path.read_text(encoding="utf-8"))
            if word_min <= chars <= word_max:
                print(f"\n字数：{chars}（在目标范围 {word_min}-{word_max} 内）", file=sys.stderr)
                finished = True
                break
            word_retries += 1
            if word_retries > word_max_retries:
                print(f"\n达到最大字数重试次数。最终：{chars} 字。", file=sys.stderr)
                finished = True
                break
            if chars < word_min:
                feedback = (
                    f"当前脚本共 {chars} 字，偏短（目标 {word_min}-{word_max} 字）。"
                    f"请继续用 edit_script 补充讲解细节，完成后重新调用 finish。"
                )
            else:
                feedback = (
                    f"当前脚本共 {chars} 字，偏长（目标 {word_min}-{word_max} 字）。"
                    f"请继续用 edit_script 精简删冗余，完成后重新调用 finish。"
                )
            print(f"\n字数：{chars}（超出范围，重试 {word_retries}/{word_max_retries}）", file=sys.stderr)
            messages.append({"role": "user", "content": feedback})
            continue

        finished = True
        break

    if not finished:
        print(f"警告：达到最大迭代次数（{max_iterations}），未收到 finish。", file=sys.stderr)


# --- Prompt builders ---

def build_draft_system_prompt(prompt_cfg, reference_block, word_min, word_max, teaching_objective):
    """System prompt for the DeepSeek draft step."""
    draft = prompt_cfg["draft"]
    guidelines = "\n".join(f"- {g}" for g in draft["voice_guidelines"])
    framework = draft["narrative_framework"]
    framework_text = "\n".join(f"  {k}：{v}" for k, v in framework.items())
    constraints = draft["constraints"]

    return (
        f"{_fill(draft['identity'], word_min, word_max, teaching_objective)}\n\n"
        f"{_fill(draft['pipeline_context'], word_min, word_max, teaching_objective)}\n\n"
        f"叙事结构（dream / expansion / mellow / overlong）：\n{framework_text}\n\n"
        f"语言风格：\n{guidelines}\n\n"
        f"要求：\n"
        f"- 字数：{_fill(constraints['length'], word_min, word_max)}\n"
        f"- 语言：{constraints['language']}\n"
        f"- 格式：{constraints['format']}\n"
        f"- 范围：{constraints['scope']}"
        f"{reference_block}"
    )


def build_fact_check_system_prompt(prompt_cfg, reference_block, word_min, word_max, teaching_objective):
    """System prompt for the Kimi fact-check step (with textbook page images)."""
    fc = prompt_cfg["fact_check"]
    constraints = fc["constraints"]

    return (
        f"{_fill(fc['identity'], word_min, word_max, teaching_objective)}\n\n"
        f"{_fill(fc['task'], word_min, word_max, teaching_objective)}\n\n"
        f"要求：\n"
        f"- 字数：{constraints['length']}\n"
        f"- 语言：{constraints['language']}\n"
        f"- 范围：{constraints['scope_note']}\n"
        f"- 格式：{constraints['format']}"
        f"{reference_block}"
    )


def build_humanize_system_prompt(prompt_cfg, reference_block, word_min, word_max, teaching_objective):
    """System prompt for the DeepSeek humanize step."""
    h = prompt_cfg["humanize"]
    constraints = h["constraints"]
    rules_text = _format_rules(h.get("自然表达准则", ""))

    return (
        f"{_fill(h['identity'], word_min, word_max, teaching_objective)}\n\n"
        f"{_fill(h['task'], word_min, word_max, teaching_objective)}\n\n"
        f"自然表达准则：\n{rules_text}\n\n"
        f"要求：\n"
        f"- 字数：{_fill(constraints['length'], word_min, word_max)}\n"
        f"- 语言：{constraints['language']}\n"
        f"- 铁律：{constraints['rule']}\n"
        f"- 格式：{constraints['format']}"
        f"{reference_block}"
    )


# --- Steps ---

def run_draft(args, project_dir, tmp_dir, prompt_cfg, reference_block,
              word_min, word_max, teaching_objective):
    """Step 1: DeepSeek writes the initial script via write_script tool."""
    analysis = load_json_file(project_dir / args.research)
    analysis_brief = format_analysis_brief(analysis)

    script_path = tmp_dir / "script.json"
    backup = tmp_dir / "script_before_draft.json"
    if script_path.exists():
        shutil.copy2(script_path, backup)

    ds_client = build_client(DEEPSEEK_CFG)

    print("\n" + "=" * 60, file=sys.stderr)
    print("STEP 1: DRAFT (DeepSeek — write_script)", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    system_prompt = build_draft_system_prompt(prompt_cfg, reference_block,
                                              word_min, word_max, teaching_objective)
    user_content = (
        f"教师教学目标：\n{teaching_objective}\n\n"
        f"以下是分析概览（编剧蓝图）：\n\n{analysis_brief}\n\n"
        f"请基于以上蓝图，调用 write_script 工具撰写完整的教学视频脚本。"
    )

    agent_loop(ds_client, DEEPSEEK_CFG, system_prompt, user_content,
               "draft", script_path)

    if script_path.exists():
        chars = count_chapter_chars(script_path.read_text(encoding="utf-8"))
        print(f"\nDraft saved: {script_path} ({chars} chars)", file=sys.stderr)
    else:
        print(f"\nWarning: {script_path} was not written.", file=sys.stderr)


def run_fact_check(project_dir, tmp_dir, prompt_cfg, reference_block,
                   word_min, word_max, teaching_objective, textbook_meta):
    """Step 2: Kimi reads script + textbook pages, fixes errors via edit_script."""
    script_path = tmp_dir / "script.json"
    if not script_path.exists():
        print(f"Error: {script_path} not found. Run --only draft first.", file=sys.stderr)
        sys.exit(1)

    backup = tmp_dir / "script_before_fact_check.json"
    shutil.copy2(script_path, backup)

    pdf_path = project_dir / "dataset" / textbook_meta["pdf_path"]
    if not pdf_path.exists():
        print(f"Error: PDF not found at {pdf_path}", file=sys.stderr)
        sys.exit(1)

    kimi_client = build_client(KIMI_CFG)

    print("\n" + "=" * 60, file=sys.stderr)
    print("STEP 2: FACT-CHECK (Kimi — read + edit)", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    pdf_images = pdf_to_base64_images(str(pdf_path), textbook_meta["page_start"], textbook_meta["page_end"])
    print(f"{len(pdf_images)} pages loaded for visual fact-check", file=sys.stderr)

    system_prompt = build_fact_check_system_prompt(prompt_cfg, reference_block,
                                                    word_min, word_max, teaching_objective)

    content = [{"type": "text", "text": "以下是教材原始页面图片（唯一的事实标准）："}]
    for img in pdf_images:
        content.append({"type": "image_url", "image_url": {"url": img}})
    content.append({
        "type": "text",
        "text": (
            "现在请核对脚本：先调用 read_script 查看当前脚本内容，"
            "然后逐条对照教材页面检查事实准确性和内容覆盖度，"
            "对每一处需要修正或补充的地方调用 edit_script，全部完成后调用 finish。"
        ),
    })

    agent_loop(kimi_client, KIMI_CFG, system_prompt, content,
               "fact_check", script_path)

    chars = count_chapter_chars(script_path.read_text(encoding="utf-8"))
    print(f"\nFact-checked script saved: {script_path} ({chars} chars)", file=sys.stderr)


def run_humanize(project_dir, tmp_dir, prompt_cfg, reference_block,
                 word_min, word_max, teaching_objective, max_retries):
    """Step 3: DeepSeek polishes voice via edit_script, lands word band."""
    script_path = tmp_dir / "script.json"
    if not script_path.exists():
        print(f"Error: {script_path} not found. Run --only fact_check first.", file=sys.stderr)
        sys.exit(1)

    backup = tmp_dir / "script_before_humanize.json"
    shutil.copy2(script_path, backup)

    ds_client = build_client(DEEPSEEK_CFG)

    print("\n" + "=" * 60, file=sys.stderr)
    print("STEP 3: HUMANIZE (DeepSeek — read + edit)", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    system_prompt = build_humanize_system_prompt(prompt_cfg, reference_block,
                                                 word_min, word_max, teaching_objective)
    user_content = (
        "请先调用 read_script 查看当前脚本，然后按自然表达准则逐处润色语气和衔接，"
        "每一处用 edit_script 修改。注意铁律：只改怎么说，不改说什么。"
        f"目标字数 {word_min}-{word_max} 字。全部完成后调用 finish。"
    )

    agent_loop(ds_client, DEEPSEEK_CFG, system_prompt, user_content,
               "humanize", script_path,
               word_check=(word_min, word_max, max_retries))


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Generate teaching video script from analysis")
    parser.add_argument("--research", default="tmp/analysis.json",
                        help="Path to stage-3 analysis JSON")
    parser.add_argument("--prompt", default="prompts/4_script.json",
                        help="Path to script prompt JSON")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Max humanize word-count retries")
    parser.add_argument("--only", choices=["draft", "fact_check", "humanize"], default=None,
                        help="Run only one step (operates on tmp/script.json)")
    args = parser.parse_args()

    project_dir = Path(__file__).parent.parent
    load_dotenv(project_dir / ".env")
    tmp_dir = project_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    prompt_cfg = load_json_file(project_dir / args.prompt)
    reference_block = build_reference_block(prompt_cfg)

    textbook_path = tmp_dir / "textbook_understanding.json"
    word_min, word_max = load_word_limits(textbook_path)
    teacher = load_teacher_meta(project_dir)
    teaching_objective = teacher["teaching_objective"]
    print(f"Word band: {word_min}-{word_max} zi", file=sys.stderr)

    if args.only in (None, "draft"):
        run_draft(args, project_dir, tmp_dir, prompt_cfg, reference_block,
                  word_min, word_max, teaching_objective)

    if args.only in (None, "fact_check"):
        run_fact_check(project_dir, tmp_dir, prompt_cfg, reference_block,
                       word_min, word_max, teaching_objective, teacher["textbook"])

    if args.only in (None, "humanize"):
        run_humanize(project_dir, tmp_dir, prompt_cfg, reference_block,
                     word_min, word_max, teaching_objective, args.max_retries)

    script_path = tmp_dir / "script.json"
    if script_path.exists():
        chars = count_chapter_chars(script_path.read_text(encoding="utf-8"))
        print(f"\nFinal script: {script_path} ({chars} chars)", file=sys.stderr)
        force_chapter_numbering(script_path)
        print("Applied 'chapter N:' numbering to chapter names.", file=sys.stderr)
        final_path = tmp_dir / "script_final.json"
        shutil.copy2(script_path, final_path)
        print(f"Copied to {final_path} for downstream stages.", file=sys.stderr)


if __name__ == "__main__":
    main()
