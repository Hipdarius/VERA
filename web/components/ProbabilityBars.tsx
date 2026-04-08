"use client";

import { motion } from "framer-motion";
import { CLASS_LABELS, type ClassProbability } from "@/lib/types";
import { useTheme } from "./ThemeProvider";

export function ProbabilityBars({
  probabilities,
  predictedClass,
}: {
  probabilities: ClassProbability[] | null;
  predictedClass: string | null;
}) {
  const { theme } = useTheme();
  const isLight = theme === "light";

  if (!probabilities) {
    return (
      <div
        className="flex h-40 items-center justify-center font-mono text-xs uppercase tracking-widest"
        style={{ color: isLight ? "#94a3b8" : "#64748b" }}
      >
        awaiting classifier\u2026
      </div>
    );
  }

  const sorted = [...probabilities].sort(
    (a, b) => b.probability - a.probability
  );

  return (
    <ul className="space-y-3">
      {sorted.map((cls) => {
        const isTop = cls.name === predictedClass;
        const pct = (cls.probability * 100).toFixed(1);

        const topColor = isLight ? "#d97706" : "#fbbf24";
        const normalColor = isLight ? "#0284c7" : "#38bdf8";
        const labelColor = isTop ? topColor : (isLight ? "#64748b" : "#94a3b8");
        const barGrad = isTop
          ? (isLight ? "linear-gradient(to right, rgba(217, 119, 6, 0.8), rgba(217, 119, 6, 0.3))" : "linear-gradient(to right, rgba(251, 191, 36, 0.8), rgba(251, 191, 36, 0.3))")
          : (isLight ? "linear-gradient(to right, rgba(2, 132, 199, 0.7), rgba(2, 132, 199, 0.1))" : "linear-gradient(to right, rgba(56, 189, 248, 0.7), rgba(56, 189, 248, 0.1))");

        return (
          <li key={cls.name}>
            <div className="mb-1 flex items-center justify-between font-mono text-[11px] uppercase tracking-widest">
              <span style={{ color: labelColor }}>
                {CLASS_LABELS[cls.name] ?? cls.name}
              </span>
              <span style={{ color: labelColor }}>
                {pct}%
              </span>
            </div>
            <div
              className="relative h-2 overflow-hidden rounded-full"
              style={{ backgroundColor: isLight ? "#e2e8f0" : "#1e293b" }}
            >
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${cls.probability * 100}%` }}
                transition={{ duration: 0.7, ease: "easeOut" }}
                className="h-full"
                style={{ backgroundImage: barGrad }}
              />
            </div>
          </li>
        );
      })}
    </ul>
  );
}
