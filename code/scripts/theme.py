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


# ── Themed base scene ────────────────────────────────────────────────
# Every generated scene inherits from ThemedScene instead of Scene, so the
# dark navy background is applied automatically — the LLM cannot forget it.
class ThemedScene(Scene):
    def setup(self):
        self.camera.background_color = BG
