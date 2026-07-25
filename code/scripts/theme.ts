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
.edu-title { font-size: var(--type-title); font-weight: 700; color: var(--fg); }
.edu-body { font-size: var(--type-body); color: var(--fg); }
.edu-label { font-size: var(--type-label); color: var(--neutral); }
.edu-small { font-size: var(--type-small); color: var(--neutral); }
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
