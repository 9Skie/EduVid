A teacher-driven, AI-assisted pipeline that turns a textbook chapter into a finished educational video. This document describes a v1 that runs and tested locally on one machine, no production infrastructure, no fallbacks, no failsafes.

---

## 1. Problem & background

The pipeline is motivated by claims about education.

**Claim 1: LLMs struggle to balance the discipline of learning against just receiving output, especially for younger kids.**

Multiple recent studies converge on _cognitive offloading_: an RCT found unrestricted-ChatGPT students scored 57.5% on a retention test vs 68.5% for traditional study; a search-as-learning study found LLMs good for building initial mental models but worse for retention than books; a middle-school field experiment found ChatGPT-guided study raised factual recall ~6% but depressed critical thinking ~8%. The freed mental effort has to be redirected into thinking, or learning suffers.

**Claim 2: LLM outputs should be reusable, not regenerated per person; conversational style makes reuse hard.**

One-to-one tutoring is hard to scale and gains attenuate when scaled, so a reusable teacher-vetted artifact is genuinely more efficient than N regenerations. But the personalization is _where the learning gain comes from_, and for video specifically, Mayer's research found a conversational style outperforms a formal one (it creates a sense of social partnership that raises engagement and effort). So "reusable" and "conversational" aren't opposites to choose between, keep the reuse win without killing interactivity.

**Claim 3: "use it or lose it", relevance and being prompted to question drive retention, not rote memorization.**

This is _retrieval practice_ / the testing effect — a century of research shows active retrieval produces large long-term-retention gains over repeated studying, robust across ages and subjects. The E=mc² intuition (history, why it was derived, what it influenced > the bare formula) maps onto _meaningful learning_ and _elaboration_. The background → concept → implications research structure is itself pedagogically sound.

**Scope decision.**

The retrieval/active-learning layer is a larger pedagogical paradigm and is explicitly _out of scope for now_. The current scope is the visual generation pipeline as a production system; the learning layer is bracketed for later.

---
## 2. The pipeline at a glance

Linear setup (Steps 0–5), a per-part inner loop (Steps 6–9), and a linear close (Step 10). Steps alternate between AI and Teacher. Every step takes an artifact in and emits an artifact out.

| Step  | Who     | Action                                                                          | Tool                                                    | Artifact                 |
| ----- | ------- | ------------------------------------------------------------------------------- | ------------------------------------------------------- | ------------------------ |
| 0     | Teacher | Proposes idea & gives relevant textbook information                             | LLM (Call)                                              | —                        |
| 1     | AI      | Read textbook → comprehensive digest                                            | LLM (Call)                                              | Textbook digest          |
| 2     | AI      | Digest → Deep research analysis result                                          | Deep research agent                                     | Research plan            |
| 3     | AI      | Digest + Deep Research → Analysis Results                                       | Deep research agent                                     | Analysis Result          |
| 3-rev | Teacher | Revise analysis results                                                         | LLM (Converse)                                          | —                        |
| 4     | AI      | Write part-divided script from analysis result + digest                         | LLM (Call)                                              | Script                   |
| 4-rev | Teacher | Revise Script                                                                   | LLM (Converse)                                          | —                        |
| 5     | AI      | Recurring characters/locations → generate cast/location references, description | LLM (Call), GPT Image 2                                 | Cast sheet, descriptions |
| 5-rev | Teacher | Revise characters/locations generations, descriptions                           | LLM (Converse), GPT Image 2                             | —                        |
| 6     | AI      | Script → Generate audio for part _n_ per line                                   | ElevenLabs                                              | Voice recordings         |
| 7     | AI      | Voice Recordings → Timestamp each line in transcript from the real audio        | ElevenLabs                                              | Timed Script             |
| 8     | AI      | Segment lines into clips + tag method + prompt                                  | LLM (Call), GPT Image 2                                 | Clip plan                |
| 9     | AI      | Run each clip's generator                                                       | LLM (Call), Seedance 2.0 / Remotion / Manim             | Part footage             |
| 9-rev | Teacher | Approve / regenerate disliked clips                                             | LLM (Converse), (Call), Seedance 2.0 / Remotion / Manim | —                        |
|       |         | Loop back to Step 6 for part _n+1_                                              |                                                         |                          |
| 10    | AI      | Composition: stitch parts + audio + captions                                    | Remotion + FFmpeg                                       | Final video              |

All model-served slots (TTS, image, video, plain LLM calls) route through **GMI Cloud**'s unified, OpenAI-compatible MaaS API to consolidate integration and billing.

The deep-research agent (Step 2 & 3) is the one exception, there is currently no good solution for what deep research agent we will use

For v1, artifacts are JSON files on disk; the teacher UI also runs locally. No database, no job queue.

---

## 3. Per-step detail: inputs, outputs, tools, and the teacher's editing power

### Step 1 — AI: Read textbook

- **In:** Job `{concept, textbook_pdf_portion, chapter_range, audience, grade_level }`.
- **Out:** Textbook digest `{ concept_intro, key_terms[], examples_used[], textbook_claims[] }`.

### Step 2 — AI: Shallow scout → research plan

- **In:** `{textbook_digest, textbook_pdf_portion}`
- **Tool:** Deep-research agent doing a light research browse.
- **Out:** Research plan, structured into parts of  `background / concept / implications`, under each part exists ideas, each idea is tagged with `{claim, source, textbook}`. The research plan is the _intent_ to research (the directions and sources to dig into), not the research itself.
	- claim is a description
	- source is a url
	- textbook is whether this idea is in the book or not, (it could be relevant, but not introduced in the book)

### Step 3 — AI: Deep research

- **In:** `{textbook_digest, textbook_pdf_portion, research_plan}`
- **Tool:** Deep-research agent doing a deep research browse.
- **Out:** Research result with three buckets: `background[]`, `concept[]`, `implications[]` , each claim carries `{claim, source, textbook}'
- The agent returns citations as metadata, which populate `source` and give the teacher review something to check against.

### Step 4 — AI: Write script

- **In:** `{textbook_digest, textbook_pdf_portion, research_result}`
- **Tool:** LLM writing the script, this will have to conform to some instructions as well
- **Out:** Script `{ chapters[] { chapter_id, title, lines[] { line_id, text } } }`.
	- The atomic unit is the line, basically a sentence. This choice propagates through the whole back half: Step 7 timestamps lines, Step 8 groups lines into clips.

### Step 5 — AI: Cast establishment

Recurring characters/locations that exist inside the script are given a video-scope wide identity, like a scientist (Marie Curie) and a place (her Paris laboratory).

- **In:** `{script, research_result}`
- **Tool:** An LLM parses through the script/research, for each found entity (they are probably real entities ok?):
	- gather real images of it
	- let the LLM look at those, write out visual_description of it
	- give image + visual_description to GPT Image 2 for generating character/location sheet
- **Out:** `{cast_set}`, set of frozen `{cast}` records.

`{cast}` is the same format the anchors use.

```
cast {
	cast_id
	is_anchor          // true for persistent defaults, elsewise not
	name
	kind               // "character" | "location"
	origin             // "real" | "fictional"
	reference_images[] // ≤ 3, used for seedance 2 & GPT Image 2
	description        // detailed bio of the character/location
}
```


--- _(per-chapter loop begins, repeat 6 → 9 for each chapter)_ ---

### Step 6 — AI: Audio Generation & Timestamp

- **In:** `{script_chapter_n}`
- **Tool:** **ElevenLabs** text-to-speech.
- **Out:** `{voice_recordings_chapter_n}`

### Step 7 — AI: Timestamp

- **In:** `{voice_recordings_chapter_n}`
- **Tool:** **ElevenLabs** returns word/line-level timestamps with the audio
- **Out:** `{timestamp_script_chapter_n}`, `{ chapter_id, audio_file, lines[] { line_id, text, start_ms, end_ms } }`. `line_id` 

### Step 8 — AI: Segment + route into clips

- **In:** `{timestamp_script_chapter_n}`
- **Tool:** LLM segment lines into clips, then classify each clip's content register, which _picks_ the method.
- **Out:** `{clip_plan}`,`{ chapter_id, clips[] { clip_id, line_ids[], start_ms, end_ms, method, prompt } }`
    - A clip covers a group of contiguous lines bounded within a single part.

Content register picks the method:

| Register                | Content                                        | Method / tool                              |
| ----------------------- | ---------------------------------------------- | ------------------------------------------ |
| Realistic / narrative   | Real-world scenes, characters, physical action | **Seedance 2.0** (+ GPT Image 2 reference) |
| Schematic / explanatory | Designed diagrams, timelines, motion graphics  | **Remotion** (React/SVG)                   |
| STEM                    | Equations, plots, geometric relationships      | **Manim** (Python)                         |

**Manim** is for STEM related subjects, except biology and chemistry.

**Remotion** is for designed visuals of all else.

### Step 9 — AI: Generate footage

- **In:** `{clip_plan, cast_set, rule_set}`
	- `{clip_plan, rule_set}` is given to manim and remotion
	- `{clip_plan, cast_set}` is given to seedance 2
- **Tools, by `method`:**
    - **Seedance 2.0** — video for realism clips. For each clip it pulls the relevant references into omni-reference: the matching **cast entries** for whichever characters the clip contains. **GPT Image 2** supplies additional reference frames / static stills as needed.
    - **Remotion** — renders React/SVG motion graphics; the AI writes the React, a local execute-and-retry harness runs it and feeds build errors back to regenerate.
    - **Manim** — renders Python math animation; same execute-and-retry harness (LLM-written Manim breaks often, so the harness catches render failures and regenerates).
- **Out:** `{chapter_footage}`, `{ chapter_id, clips[] { clip_id, video_file, duration_ms } }`. `clip_id` links plan → footage.

--- _(loop until all parts done)_ ---

### Step 10 — AI: Composition

- **In:** `{all_chapter_footage, voice_recordings_chapters_all}`
- **Tools:** 
	- **Remotion** drives the timeline (laying clips against the measured timestamps).
	- **FFmpeg** muxes/concatenates/normalizes the heterogeneous clips (Seedance MP4s, Manim MP4s, Remotion output) + the audio track into one final encode. 
	- **Captions** are emitted directly from the timed transcript (`text + start_ms + end_ms` → `.vtt`/ `.srt`)
- **Out:** Final video + caption track.

---

The harness rests on one principle: **every tool stays consistent by reusing a frozen artifact in its own native medium.** Seedance reuses frozen _images_; Manim and Remotion reuse frozen _code_. One idea, three implementations. There is no drift-management machinery in v1 — consistency is structural (you reuse the asset), not hoped-for (you ask a prompt nicely). A multimodal drift-checker is _optional polish, deferred_; the teacher catches off-model clips at Step 9-rev anyway.

These frozen artifacts are the project-wide persistent assets: the three default anchor characters, the Manim `theme.py`, and the Remotion theme. The per-video cast (Marie Curie, her lab) is built and frozen fresh at Step 5; everything else here is fixed up front.

### Seedance — frozen reference images

Three default anchor characters (`is_anchor: true`), plus per-video content-derived cast. Each clip pulls the relevant `cast` reference images into omni-reference, so the same image goes in every time — consistency is structural.

```
CastEntry {
	cast_id: "teacher"
	is_anchor: true
	name: "Teacher"
	kind: "character"  | "location"
  reference_images: [ ... ]      // ≤3 — GPT Image 2 Refer to Teacher's Selfie, or pure GPT Image 2
  description: "The host and constant of every video — the teacher at the center of the teaching framework, present across all content as the recurring narrator-presence the students orient around. Warm, approachable, authoritative. Rendered in the project art style. If origin is 'real', this is a stylized avatar of the actual teacher, anchored to their own photo so the host carries their real likeness."
}
```

### Manim — a frozen `theme.py`

A hand-authored module every generated scene is **required to import and use** — not a prose spec the model may ignore. The look lives in code it reuses, so it can't drift.


```python
# theme.py — frozen Manim style library. Every generated scene imports from here
# and is REQUIRED to use these constants and helpers instead of hand-rolling its own.
from manim import *

# ── Palette (semantic roles, exact hex) ──────────────────────────────
PRIMARY    = "#2D5BFF"   # main objects, primary lines
ACCENT     = "#FF8A3D"   # secondary emphasis
HIGHLIGHT  = "#FFD23F"   # the thing the viewer should look at right now
CORRECT    = "#34C759"   # right answers, success states
ERROR      = "#FF3B30"   # wrong answers, error states
NEUTRAL    = "#8E8E93"   # de-emphasized / supporting elements
BG         = "#0E1116"   # scene background
FG         = "#F2F2F7"   # default text / foreground

# ── Type scale ───────────────────────────────────────────────────────
FONT        = "Inter"
SIZE_TITLE  = 54
SIZE_BODY   = 36
SIZE_LABEL  = 28
SIZE_SMALL  = 22

# ── Layout ───────────────────────────────────────────────────────────
SAFE_MARGIN  = 0.6   # min distance from frame edge (Manim units)
SPACING_UNIT = 0.4   # base gap between elements
TITLE_POS    = UP * 3.2

# ── Motion presets ───────────────────────────────────────────────────
ENTER_TIME    = 0.6
EMPHASIS_TIME = 0.4
EASE          = smooth   # default rate_func

# ── Styled helpers (call these instead of raw Mobjects) ──────────────
def TitleText(s):
    return Text(s, font=FONT, font_size=SIZE_TITLE, color=FG, weight=BOLD).to_edge(UP, buff=SAFE_MARGIN)

def BodyText(s):
    return Text(s, font=FONT, font_size=SIZE_BODY, color=FG)

def Label(s):
    return Text(s, font=FONT, font_size=SIZE_LABEL, color=NEUTRAL)

def Eq(latex):                       # equations use the FG color by default
    return MathTex(latex, color=FG)

def StyledArrow(start, end, color=PRIMARY):
    return Arrow(start, end, color=color, stroke_width=6, buff=0.1, tip_length=0.25)

def Highlight(mobj):                  # standard "look here" emphasis
    return Indicate(mobj, color=HIGHLIGHT, scale_factor=1.15, run_time=EMPHASIS_TIME)

def enter(mobj):                      # standard entrance
    return FadeIn(mobj, shift=UP*0.3, run_time=ENTER_TIME, rate_func=EASE)
```

### Remotion — a frozen theme + component set

The web-native equivalent: the same palette, type roles, spacing, and entrance motion, as CSS-variable tokens plus reusable components every composition imports and composes from (never bare JSX with ad-hoc CSS).

```tsx
// theme.ts — frozen Remotion style tokens. Generated code composes from the styled
// components below, never raw divs with ad-hoc CSS.

export const COLORS = {
  primary:   "#2D5BFF",
  accent:    "#FF8A3D",
  highlight: "#FFD23F",
  correct:   "#34C759",
  error:     "#FF3B30",
  neutral:   "#8E8E93",
  bg:        "#0E1116",
  fg:        "#F2F2F7",
};

export const FONT = "Inter";
export const TYPE = { title: 72, body: 40, label: 30, small: 24 };  // px
export const SPACE = { unit: 16, safeMargin: 64 };                  // px
export const MOTION = { enterFrames: 18, emphasisFrames: 12 };      // at 30fps

// ── Base components (generated compositions use these, not bare JSX) ──
import { interpolate, useCurrentFrame } from "remotion";

export const Title: React.FC<{children: React.ReactNode}> = ({children}) => (
  <div style={{ fontFamily: FONT, fontSize: TYPE.title, fontWeight: 700,
                color: COLORS.fg, position: "absolute", top: SPACE.safeMargin,
                left: SPACE.safeMargin }}>{children}</div>
);

export const Body: React.FC<{children: React.ReactNode}> = ({children}) => (
  <div style={{ fontFamily: FONT, fontSize: TYPE.body, color: COLORS.fg }}>{children}</div>
);

export const Label: React.FC<{children: React.ReactNode}> = ({children}) => (
  <div style={{ fontFamily: FONT, fontSize: TYPE.label, color: COLORS.neutral }}>{children}</div>
);

export const Card: React.FC<{children: React.ReactNode}> = ({children}) => (
  <div style={{ background: "#161B22", borderRadius: 16, padding: SPACE.unit * 2,
                margin: SPACE.unit, color: COLORS.fg }}>{children}</div>
);

// standard entrance — fade + slide, used everywhere for a consistent reveal
export const FadeUp: React.FC<{children: React.ReactNode}> = ({children}) => {
  const frame = useCurrentFrame();
  const opacity = interpolate(frame, [0, MOTION.enterFrames], [0, 1], { extrapolateRight: "clamp" });
  const y = interpolate(frame, [0, MOTION.enterFrames], [12, 0], { extrapolateRight: "clamp" });
  return <div style={{ opacity, transform: `translateY(${y}px)` }}>{children}</div>;
};
```

The Manim and Remotion themes are deliberately the **same design system in two languages** — identical palette, matching type roles, same spacing logic, matching entrance motion — so a Manim clip and a Remotion clip in the same video read as one consistent visual family. The hex values, font, and sizes are starter defaults, tune to taste; the _structure_ (reuse a frozen library) is what enforces consistency.

---

