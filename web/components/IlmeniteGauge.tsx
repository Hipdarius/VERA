"use client";

import { motion } from "framer-motion";

export function IlmeniteGauge({
  fraction,
  trueFraction,
}: {
  fraction: number | null;
  trueFraction?: number | null;
}) {
  const f = fraction ?? 0;
  const pct = Math.max(0, Math.min(1, f)) * 100;

  return (
    <div className="flex flex-col items-center gap-3">
      <div className="relative h-40 w-40">
        {/* Background ring */}
        <svg viewBox="0 0 120 120" className="absolute inset-0">
          <circle
            cx="60"
            cy="60"
            r="52"
            fill="none"
            stroke="#1f2533"
            strokeWidth="10"
          />
          <motion.circle
            cx="60"
            cy="60"
            r="52"
            fill="none"
            stroke="url(#ilmGrad)"
            strokeWidth="10"
            strokeLinecap="round"
            strokeDasharray={2 * Math.PI * 52}
            initial={{ strokeDashoffset: 2 * Math.PI * 52 }}
            animate={{
              strokeDashoffset: 2 * Math.PI * 52 * (1 - pct / 100),
            }}
            transition={{ duration: 0.9, ease: "easeOut" }}
            transform="rotate(-90 60 60)"
          />
          <defs>
            <linearGradient id="ilmGrad" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor="#22d3ee" />
              <stop offset="100%" stopColor="#fbbf24" />
            </linearGradient>
          </defs>
        </svg>

        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="font-mono text-3xl font-semibold text-amber-glow">
            {fraction !== null ? `${pct.toFixed(1)}%` : "—"}
          </span>
          <span className="font-mono text-[9px] uppercase tracking-widest text-slate-500">
            ilmenite mass
          </span>
        </div>
      </div>

      {trueFraction !== undefined && trueFraction !== null && (
        <div className="font-mono text-[10px] uppercase tracking-widest text-slate-500">
          ground truth · {(trueFraction * 100).toFixed(1)}%
        </div>
      )}
    </div>
  );
}
