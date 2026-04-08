"use client";

import { motion } from "framer-motion";
import { useTheme } from "./ThemeProvider";

export function IlmeniteGauge({
  fraction,
  trueFraction,
}: {
  fraction: number | null;
  trueFraction?: number | null;
}) {
  const { theme } = useTheme();
  const isLight = theme === "light";

  const f = fraction ?? 0;
  const pct = Math.max(0, Math.min(1, f)) * 100;

  const ringBg = isLight ? "#e2e8f0" : "#1e293b";
  const valueColor = isLight ? "#7c3aed" : "#8b5cf6";
  const mutedColor = isLight ? "#94a3b8" : "#64748b";

  return (
    <div className="flex flex-col items-center gap-3">
      <div className="relative h-40 w-40">
        <svg viewBox="0 0 120 120" className="absolute inset-0">
          <circle cx="60" cy="60" r="52" fill="none" stroke={ringBg} strokeWidth="10" />
          <motion.circle
            cx="60"
            cy="60"
            r="52"
            fill="none"
            stroke={isLight ? "url(#ilmGradLight)" : "url(#ilmGrad)"}
            strokeWidth="10"
            strokeLinecap="round"
            strokeDasharray={2 * Math.PI * 52}
            initial={{ strokeDashoffset: 2 * Math.PI * 52 }}
            animate={{ strokeDashoffset: 2 * Math.PI * 52 * (1 - pct / 100) }}
            transition={{ duration: 0.9, ease: "easeOut" }}
            transform="rotate(-90 60 60)"
          />
          <defs>
            <linearGradient id="ilmGrad" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor="#38bdf8" />
              <stop offset="100%" stopColor="#8b5cf6" />
            </linearGradient>
            <linearGradient id="ilmGradLight" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor="#0284c7" />
              <stop offset="100%" stopColor="#7c3aed" />
            </linearGradient>
          </defs>
        </svg>

        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="font-mono text-3xl font-semibold" style={{ color: valueColor }}>
            {fraction !== null ? `${pct.toFixed(1)}%` : "\u2014"}
          </span>
          <span className="font-mono text-[9px] uppercase tracking-widest" style={{ color: mutedColor }}>
            ilmenite mass
          </span>
        </div>
      </div>

      {trueFraction !== undefined && trueFraction !== null && (
        <div className="font-mono text-[10px] uppercase tracking-widest" style={{ color: mutedColor }}>
          ground truth \u00B7 {(trueFraction * 100).toFixed(1)}%
        </div>
      )}
    </div>
  );
}
