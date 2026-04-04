"use client";

import { motion } from "framer-motion";
import { CLASS_LABELS, type ClassProbability } from "@/lib/types";

export function ProbabilityBars({
  probabilities,
  predictedClass,
}: {
  probabilities: ClassProbability[] | null;
  predictedClass: string | null;
}) {
  if (!probabilities) {
    return (
      <div className="flex h-40 items-center justify-center font-mono text-xs uppercase tracking-widest text-slate-500">
        awaiting classifier…
      </div>
    );
  }

  // Sort descending so the most probable class always renders on top.
  const sorted = [...probabilities].sort(
    (a, b) => b.probability - a.probability
  );

  return (
    <ul className="space-y-3">
      {sorted.map((cls) => {
        const isTop = cls.name === predictedClass;
        const pct = (cls.probability * 100).toFixed(1);
        const tone = isTop
          ? "from-amber-glow/80 to-amber-glow/30"
          : "from-cyan-glow/70 to-cyan-glow/10";
        return (
          <li key={cls.name}>
            <div className="mb-1 flex items-center justify-between font-mono text-[11px] uppercase tracking-widest">
              <span className={isTop ? "text-amber-glow" : "text-slate-400"}>
                {CLASS_LABELS[cls.name] ?? cls.name}
              </span>
              <span
                className={isTop ? "text-amber-glow" : "text-slate-400"}
              >
                {pct}%
              </span>
            </div>
            <div className="relative h-2 overflow-hidden rounded-full bg-void-700">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${cls.probability * 100}%` }}
                transition={{ duration: 0.7, ease: "easeOut" }}
                className={`h-full bg-gradient-to-r ${tone}`}
              />
            </div>
          </li>
        );
      })}
    </ul>
  );
}
