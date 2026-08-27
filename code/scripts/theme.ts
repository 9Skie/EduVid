// Frozen EduVid motion theme. 3_clips.py injects THEME_CSS into every motion
// clip HTML wrapper. The LLM does not redefine these values.
export const THEME_CSS = `
:root {
  --bg: #0E1116;
  --fg: #F2F2F7;
  --primary: #2D5BFF;
  --accent: #FF8A3D;
  --highlight: #FFD23F;
  --correct: #34C759;
  --error: #FF3B30;
  --neutral: #8E8E93;
  --card: #161B22;
  --font: Inter, -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", sans-serif;
  --type-title: 54px;
  --type-body: 36px;
  --type-label: 28px;
  --type-small: 22px;
  --space-unit: 16px;
  --safe-margin: 64px;
  --radius: 16px;
  --motion-enter: 600ms;
  --motion-emphasis: 400ms;
  --ease: cubic-bezier(.2,.8,.2,1);
}
html, body {
  margin: 0;
  width: 1280px;
  height: 720px;
  overflow: hidden;
  background: var(--bg);
  color: var(--fg);
  font-family: var(--font);
}
#stage {
  position: relative;
  width: 1280px;
  height: 720px;
  overflow: hidden;
  background: var(--bg);
}
/* SVG paint bridge. Generated scenes are usually a single inline <svg>, and SVG
   ignores `color` unless paint is routed through currentColor. Defaulting fill to
   currentColor makes every role class below work identically on HTML and on SVG,
   so the model can pick a semantic class instead of hard-coding a hex. */
#stage svg { display: block; }
#stage svg text { font-family: var(--font); }
/* Default the paint to currentColor only where the author has NOT set fill themselves.
   Wrapped in :where() so this carries zero specificity: an explicit fill="..." attribute
   is excluded outright, and the role classes below (.edu-primary, .edu-outline, ...)
   still win. Setting fill unconditionally here would silently repaint every author
   colour -- black text on a light card would turn near-white and vanish. */
:where(#stage svg:not([fill]), #stage svg *:not([fill])) { fill: currentColor; }

.edu-title { font-size: var(--type-title); font-weight: 700; color: var(--fg); }
.edu-body { font-size: var(--type-body); color: var(--fg); }
.edu-label { font-size: var(--type-label); color: var(--neutral); }
.edu-small { font-size: var(--type-small); color: var(--neutral); }

/* Semantic paint roles. Use these instead of literal colours; they set `color` so
   HTML text, SVG fill and `stroke: currentColor` all follow the same token. */
.edu-fg { color: var(--fg); }
.edu-muted { color: var(--neutral); }
.edu-primary { color: var(--primary); }
.edu-accent { color: var(--accent); }
.edu-highlight { color: var(--highlight); }
.edu-correct { color: var(--correct); }
.edu-error { color: var(--error); }
/* Outline variants: paint the stroke, leave the interior empty. */
.edu-outline { fill: none; stroke: currentColor; stroke-width: 4; stroke-linejoin: round; stroke-linecap: round; }
.edu-hairline { fill: none; stroke: currentColor; stroke-width: 2; }
.edu-card {
  background: var(--card);
  border-radius: var(--radius);
  padding: calc(var(--space-unit) * 2);
  color: var(--fg);
}
.edu-enter { animation: eduFadeUp var(--motion-enter) var(--ease) both; }
.edu-emphasis { animation: eduPulse var(--motion-emphasis) var(--ease) both; }
@keyframes eduFadeUp {
  from { opacity: 0; transform: translateY(12px); }
  to { opacity: 1; transform: translateY(0); }
}
@keyframes eduPulse {
  0% { transform: scale(1); }
  50% { transform: scale(1.04); }
  100% { transform: scale(1); }
}
`;
