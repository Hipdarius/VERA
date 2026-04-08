"use client";

import { motion } from "framer-motion";
import { useTheme } from "./ThemeProvider";

export function Hero({
  schemaVersion,
  modelLoaded,
  metaLoading = false,
}: {
  schemaVersion: string | null;
  modelLoaded: boolean;
  metaLoading?: boolean;
}) {
  const { theme, toggle } = useTheme();

  return (
    <header className="relative overflow-hidden border-b border-cyan-glow/15 dark:border-cyan-glow/15 light:border-slate-200 bg-gradient-to-b from-void-900 via-void-800 to-void-900 dark:from-void-900 dark:via-void-800 dark:to-void-900 px-6 py-12 sm:py-16"
      style={{
        background: theme === "light"
          ? "linear-gradient(to bottom, #f8fafc, #f1f5f9, #f8fafc)"
          : undefined,
        borderColor: theme === "light" ? "#e2e8f0" : undefined,
      }}
    >
      <div className="mx-auto flex max-w-6xl flex-col gap-6">
        <div className="flex items-center justify-between">
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="flex items-center gap-3 text-xs font-mono uppercase tracking-[0.3em]"
            style={{ color: theme === "light" ? "#0369a1" : "rgba(56, 189, 248, 0.8)" }}
          >
            <span
              className="inline-block h-2 w-2 animate-pulse-soft rounded-full"
              style={{
                backgroundColor: theme === "light" ? "#0284c7" : "#38bdf8",
                boxShadow: theme === "light" ? "0 0 12px rgba(2, 132, 199, 0.4)" : "0 0 24px rgba(56, 189, 248, 0.35)",
              }}
            />
            Mission Control · Lunar Surface Operations
          </motion.div>

          <button
            onClick={toggle}
            className="flex items-center gap-2 rounded-full border px-3 py-1.5 font-mono text-[10px] uppercase tracking-wider transition-all"
            style={{
              borderColor: theme === "light" ? "#cbd5e1" : "rgba(56, 189, 248, 0.3)",
              background: theme === "light" ? "rgba(255,255,255,0.8)" : "rgba(15, 23, 42, 0.6)",
              color: theme === "light" ? "#475569" : "#94a3b8",
            }}
          >
            {theme === "dark" ? (
              <>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>
                Light
              </>
            ) : (
              <>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
                Dark
              </>
            )}
          </button>
        </div>

        <motion.h1
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.05 }}
          className="font-mono text-4xl font-semibold leading-tight sm:text-6xl"
        >
          <span
            className="bg-clip-text text-transparent"
            style={{
              backgroundImage: theme === "light"
                ? "linear-gradient(to right, #0284c7, #0f172a, #7c3aed)"
                : "linear-gradient(to right, #38bdf8, #f1f5f9, #8b5cf6)",
            }}
          >
            VERA
          </span>
          <span
            className="block text-base font-normal tracking-wider sm:text-lg"
            style={{ color: theme === "light" ? "#64748b" : "#94a3b8" }}
          >
            VIS/NIR + 405 nm LIF mineral classification probe
          </span>
        </motion.h1>

        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.7, delay: 0.15 }}
          className="max-w-2xl text-sm leading-relaxed sm:text-base"
          style={{ color: theme === "light" ? "#64748b" : "#94a3b8" }}
        >
          A compact spectrometer that fingerprints lunar regolith in real time.
          Click{" "}
          <span style={{ color: theme === "light" ? "#0284c7" : "#38bdf8" }}>
            Initiate Scan
          </span>{" "}
          to fire a synthetic acquisition through the trained 1D&nbsp;ResNet and
          read out a mineral class plus ilmenite mass fraction in milliseconds.
        </motion.p>

        <div className="flex flex-wrap gap-3 font-mono text-[10px] uppercase tracking-wider">
          {metaLoading ? (
            <Pill label="status" value="connecting..." ok={false} theme={theme} />
          ) : (
            <>
              <Pill label="schema" value={schemaVersion ?? "?"} ok={!!schemaVersion} theme={theme} />
              <Pill label="model" value={modelLoaded ? "ONNX online" : "offline"} ok={modelLoaded} theme={theme} />
              <Pill label="runtime" value="onnxruntime" ok theme={theme} />
            </>
          )}
        </div>
      </div>
    </header>
  );
}

function Pill({ label, value, ok, theme }: { label: string; value: string; ok: boolean; theme: string }) {
  const isLight = theme === "light";
  const okColor = isLight ? "#0284c7" : "#34d399";
  const warnColor = isLight ? "#f59e0b" : "#f59e0b";
  const color = ok ? okColor : warnColor;

  return (
    <div
      className="flex items-center gap-2 rounded-full border px-3 py-1"
      style={{
        borderColor: ok
          ? (isLight ? "rgba(2, 132, 199, 0.3)" : "rgba(52, 211, 153, 0.3)")
          : (isLight ? "rgba(245, 158, 11, 0.3)" : "rgba(245, 158, 11, 0.3)"),
        background: isLight ? "rgba(255,255,255,0.6)" : "rgba(15, 23, 42, 0.6)",
        color,
        boxShadow: isLight ? "none" : (ok ? "0 0 24px rgba(52, 211, 153, 0.35)" : "0 0 24px rgba(245, 158, 11, 0.35)"),
      }}
    >
      <span style={{ color: isLight ? "#94a3b8" : "#64748b" }}>{label}</span>
      <span>{value}</span>
    </div>
  );
}
