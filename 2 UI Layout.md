# Teacher-Facing UI — Layout Spec (ASCII)

Base is two panes: **step rail** (left) + **main workspace** (center). A **read-only
advisor** opens as a **third column on the right** when the teacher summons it — the
workspace shrinks to make room; it is not a floating overlay. The center is the producing
agent + the artifact being edited; the advisor reads the current artifact and advises. The
teacher is the only one who actually writes the artifact.

---

## 1. Overall frame

Two columns by default. Opening the advisor (✦) splits the right into a third column and
narrows the workspace; closing it returns the workspace to full width.

**Advisor open — three columns:**

```
┌──────────┬─────────────────────────────────┬───────────────────┐
│  STEPS   │          MAIN WORKSPACE    [ ✦ ] │   ADVISOR         │
│  (rail)  │   (producing agent + artifact)   │   (read-only)     │
│          │                                  │                   │
│  ● 2     │                                  │  reads current    │
│  ○ 3     │                                  │  artifact;        │
│  ○ …     │                                  │  advises only;    │
│          │                                  │  cannot write     │
│          │                                  │  [ × close ]      │
└──────────┴─────────────────────────────────┴───────────────────┘
   ~15%                 ~60%                          ~25%
```

**Advisor closed — two columns, workspace full width:**

```
┌──────────┬───────────────────────────────────────────────────────┐
│  STEPS   │                 MAIN WORKSPACE                  [ ✦ ]  │
│  (rail)  │          (producing agent + live artifact)            │
└──────────┴───────────────────────────────────────────────────────┘
   ~15%                          ~85%
```

---

## 2. Left rail — step nodes

Nodes stacked top-to-bottom. Current = ●, done = ✓, locked-ahead = ○. Each AI step is
followed by its teacher review (`-rev`). The rail shows where you are and what's frozen
behind you (freeze-on-handoff). The per-chapter steps (6–9) nest under a chapter header
during the loop.

```
┌────────────┐
│   STEPS    │
├────────────┤
│ ✓ 1  Read  │  (locked)
│ ● 2  Research│            ◀ HERE
│ ○ 3  Analysis│ 
│ ○ 3-rev    │  
│ ○ 4  Script│
│ ○ 4-rev    │
│ ○ 5  Cast  │  
│ ○ 5-rev    │
│ ┌─ Chapter n ─┐
│ ○ 6  Audio  │
│ ○ 7  Time   │
│ ○ 6,7-rev   │
│ ○ 8  Clips  │  segment + route
│ ○ 8-rev     │
│ ○ 9  Gen    │  generate footage
│ ○ 9-rev     │  approve / regen
│ └───────────┘
│ ○ 10 Final │  composition
└────────────┘
```

Locked steps are viewable but not editable — clicking one shows its frozen artifact
read-only. You cannot reach back past your current window (forward-only).

---

## 3. Center — the main workspace (varies by step)

The center holds the **producing agent's output as a document-like artifact**, plus the
controls to edit/confirm and advance. It is a big workspace, not a tiny chat — the agent
emits comprehensive documents, and the teacher edits them in place.

### 3a. Intake (Step 1 entry) — the "big empty chat" first contact

```
┌──────────────────────────────────────────────┐
│  MAIN WORKSPACE — New video                    │
│                                                │
│   What do you want to teach?                   │
│   ┌──────────────────────────────────────┐     │
│   │ y = ax + b, for elementary students  │     │
│   └──────────────────────────────────────┘     │
│                                                │
│   Textbook:  [ + attach PDF ]  chapters [1–2]  │
│   Audience:  grade [____]  reading level [__]  │
│                                                │
│                          [  Begin  ▶  ]        │
└──────────────────────────────────────────────┘
```

### 3b. Step 1 — textbook understanding

The agent reads the textbook and emits a **digest**. No teacher review gate here —
the digest flows straight into research.

```
┌──────────────────────────────────────────────┐
│  STEP 1 — Textbook understanding   [editable] │
│  ───────────────────────────────────────────  │
│  Concept intro                                 │
│   • A line y = ax + b: a is the slope, b …     │
│  Key terms                                     │
│   • slope (a) … • intercept (b) …              │
│  Examples the book uses                        │
│   • p.14 fig 2: doubling recipe …              │
│  Claims  (✎ each is editable; src-tagged)      │
│   • [p.12] "a controls steepness"   ✎          │
│   • [p.13] "b is where the line …"  ✎          │
│                                                │
│   Looks right?   [ Edit ]   [ Confirm ▶ ]      │
└──────────────────────────────────────────────┘
```

### 3c. Step 2 + 3 — two-phase deep research

**Step 2 — shallow scout → research PLAN (not the research).** The agent does a light
browse and reports *what it intends to dig into*, for the teacher to shape.

```
┌──────────────────────────────────────────────┐
│  STEP 2 — Proposed research plan   [editable]  │
│  ───────────────────────────────────────────  │
│  From a shallow look, I plan to dig into:      │
│   1. ✎ Descartes — coordinate geometry origin │
│        (textbook ✓)                            │
│   2. ✎ "slope-intercept" form, history of     │
│        (online — credible, not in book)        │
│   3. ✎ where linear models show up later …    │
│      grouped: background ▸ concept ▸ implic.   │
│                                                │
│   + add a direction   – remove                 │
│   [ Edit plan ]      [ Run deep research ▶ ]   │
└──────────────────────────────────────────────┘
```

**Step 3 — deep dive runs (async), then returns the cited research result.**

```
┌──────────────────────────────────────────────┐
│  STEP 3 — Researching… (this takes a while)    │
│   ▰▰▰▰▰▱▱▱  searching • reading • synthesizing  │
│                                                │
│  → returns: Research result                     │
│     background[] / concept[] / implications[]   │
│     each claim: { claim, source, in-textbook? } │
└──────────────────────────────────────────────┘
```

### 3d. Step 5 — cast establishment

A card grid of the recurring real characters/locations the agent found, each with its
generated reference and editable bio. The teacher prunes, adds, regenerates, edits.

```
┌──────────────────────────────────────────────┐
│  STEP 5 — Cast   [editable]                    │
│  ───────────────────────────────────────────  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐        │
│  │ [img]    │ │ [img]    │ │ + add    │        │
│  │ Marie C. │ │ Paris lab│ │          │        │
│  │ ✎ bio    │ │ ✎ bio    │ │          │        │
│  │ regen ↻  │ │ regen ↻  │ │          │        │
│  └──────────┘ └──────────┘ └──────────┘        │
│  anchors (teacher · John · Mary) shown locked  │
│                       [ Edit ]   [ Confirm ▶ ] │
└──────────────────────────────────────────────┘
```

### 3e. Generic doc-edit review (Steps 3-rev, 4-rev)

The text-document reviews (research result, script) share this shape: artifact in the
middle, per-item edit, a confirm that advances and *freezes* the artifact.

```
┌──────────────────────────────────────────────┐
│  STEP n-rev — <artifact name>      [editable]  │
│  ───────────────────────────────────────────  │
│   <items, each individually editable ✎>        │
│   ……                                           │
│   ⚠ confirming locks this layer.               │
│                       [ Edit ]   [ Confirm ▶ ] │
└──────────────────────────────────────────────┘
```

The back-half steps (6/7, 8, 9, 10) are *not* doc edits — they are specialized surfaces
(audio, clip planning, video review, final cut), each below.

### 3f. Steps 6/7 + 6,7-rev — audio + timing (per chapter)

An audio surface, not a document. Each line plays back with its measured timing; the
teacher re-records or adjusts tone/speed/energy per line. Editing a line reflows the
timeline. The script text itself is locked here.

```
┌──────────────────────────────────────────────┐
│  STEP 6/7 — Audio & timing · Chapter 2  [edit] │
│  ───────────────────────────────────────────  │
│  ▸ line 1  "Let's start with the slope…"       │
│    ▷ ▮▮▮▮▮▯▯  0:00–0:04   ↻ re-record  ✎ tone  │
│  ▸ line 2  "Watch what happens when a grows."  │
│    ▷ ▮▮▮▯▯▯▯  0:04–0:07   ↻ re-record  ✎ tone  │
│  ▸ line 3  "Now b shifts the whole line up."   │
│    ▷ ▮▮▮▮▯▯▯  0:07–0:10   ↻ re-record  ✎ tone  │
│                                                │
│  tone · speed · energy per line  (script locked)│
│  ⚠ confirming locks audio + timing.            │
│                       [ Edit ]   [ Confirm ▶ ] │
└──────────────────────────────────────────────┘
```

### 3g. Step 8 + 8-rev — clip plan (segment + route)

A grouped view: lines bundled into clips, each clip tagged with its method
(Seedance/Remotion/Manim) and its prompt. The teacher re-groups lines, overrides the
method, and edits the prompt per clip.

```
┌──────────────────────────────────────────────┐
│  STEP 8 — Clip plan · Chapter 2     [editable] │
│  ───────────────────────────────────────────  │
│  ┌ Clip 1 · lines 1–2 · 0:00–0:07 ───────────┐ │
│  │ method: [ Manim ▾ ]                        │ │
│  │ prompt: ✎ "plot y=ax+b, sweep a 1→3,       │ │
│  │            line steepens"                  │ │
│  └────────────────────────────────────────────┘ │
│  ┌ Clip 2 · line 3 · 0:07–0:10 ──────────────┐ │
│  │ method: [ Remotion ▾ ]                     │ │
│  │ prompt: ✎ "label b as the y-intercept,     │ │
│  │            arrow to crossing point"        │ │
│  └────────────────────────────────────────────┘ │
│  ↕ drag lines between clips   + split   ⨯ merge │
│  ⚠ confirming locks the plan.                  │
│                       [ Edit ]   [ Confirm ▶ ] │
└──────────────────────────────────────────────┘
```

### 3h. Step 9 + 9-rev — footage review (per chapter)

A video-review surface: each generated clip plays; the teacher approves it or regenerates
with an edited prompt. A length-mismatch flag shows when footage is shorter than its audio
target.

```
┌──────────────────────────────────────────────┐
│  STEP 9 — Footage · Chapter 2       [review]   │
│  ───────────────────────────────────────────  │
│  ┌ Clip 1 (Manim) ──────────────┐  ✓ ok        │
│  │ ▷ [ video preview ]          │  0:07 / 0:07 │
│  │                              │  [ approve ] │
│  └──────────────────────────────┘  [ regen ↻ ] │
│  ┌ Clip 2 (Remotion) ───────────┐  ⚠ short      │
│  │ ▷ [ video preview ]          │  0:02 / 0:03 │
│  │                              │  [ approve ] │
│  │ ✎ tweak prompt → regen       │  [ regen ↻ ] │
│  └──────────────────────────────┘              │
│  ⚠ confirming locks this chapter; advance.     │
│                  [ Regen selected ] [ Confirm ▶]│
└──────────────────────────────────────────────┘
```

### 3i. Step 10 — final composition

The assembled cut: full timeline of all chapters' clips against the audio, captions
toggleable. Largely a preview — the chapters are already approved — with a final play and
export.

```
┌──────────────────────────────────────────────┐
│  STEP 10 — Final video                         │
│  ───────────────────────────────────────────  │
│        ┌──────────────────────────────┐        │
│        │      ▷  [ full preview ]      │        │
│        └──────────────────────────────┘        │
│  timeline:                                     │
│  Ch1 ▮▮▮▮│Ch2 ▮▮▮▮▮▮│Ch3 ▮▮▮▮▮  ◀ playhead     │
│  audio  ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬               │
│  captions [ on ▾ ]                             │
│                                                │
│                     [ Play ]   [ Export ⬇ ]    │
└──────────────────────────────────────────────┘
```

---

## 4. Right — the advisor column (read-only)

Opened by the ✦ button, it takes the right third of the screen (the workspace shrinks to
fit). Pinned to whatever artifact the teacher is *currently* editing, it can read the
artifact and talk; it **cannot write it**. Its job: consolidate, fact-check, unstick a
confused teacher, suggest — never edit. Closing it returns the workspace to full width.

```
┌────────────────────────┐
│  ✦ ADVISOR (read-only) │
│  reading: Step 2 plan  │  [ × ]
├────────────────────────┤
│ ◦ "Descartes is right  │
│   for coordinate geo,  │
│   but slope-intercept  │
│   form is later — want │
│   me to flag that      │
│   split?"              │
│                        │
│ you: ┌──────────────┐  │
│      │ is #2 ok?    │  │
│      └──────────────┘  │
│                        │
│  advises only — you    │
│  make edits in the     │
│  center workspace      │
└────────────────────────┘
```