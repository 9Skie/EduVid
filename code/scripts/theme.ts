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
import React from "react";

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

// ── Themed background wrapper ────────────────────────────────────────
// Every composition's root wraps its content in <ThemedBackground> so the
// dark navy background is guaranteed without the LLM needing to set it.
export const ThemedBackground: React.FC<{children: React.ReactNode}> = ({children}) => (
  <div style={{ background: COLORS.bg, width: "100%", height: "100%",
                position: "absolute", top: 0, left: 0 }}>{children}</div>
);
