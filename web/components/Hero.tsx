"use client";

import { motion } from "framer-motion";

export function Hero({
  schemaVersion,
  modelLoaded,
}: {
  schemaVersion: string | null;
  modelLoaded: boolean;
}) {
  return (
    <header className="relative overflow-hidden border-b border-cyan-glow/15 bg-gradient-to-b from-void-900 via-void-800 to-void-900 px-6 py-12 sm:py-16">
      <div className="mx-auto flex max-w-6xl flex-col gap-6">
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="flex items-center gap-3 text-xs font-mono uppercase tracking-[0.3em] text-cyan-glow/80"
        >
          <span className="inline-block h-2 w-2 animate-pulse-soft rounded-full bg-cyan-glow shadow-glow-cyan" />
          Mission Control · Lunar Surface Operations
        </motion.div>

        <motion.h1
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.05 }}
          className="font-mono text-4xl font-semibold leading-tight text-slate-100 sm:text-6xl"
        >
          <span className="bg-gradient-to-r from-cyan-glow via-slate-100 to-amber-glow bg-clip-text text-transparent">
            REGOSCAN
          </span>
          <span className="block text-base font-normal tracking-wider text-slate-400 sm:text-lg">
            VIS/NIR + 405 nm LIF mineral classification probe
          </span>
        </motion.h1>

        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.7, delay: 0.15 }}
          className="max-w-2xl text-sm leading-relaxed text-slate-400 sm:text-base"
        >
          A compact spectrometer that fingerprints lunar regolith in real time.
          Click <span className="text-cyan-glow">Initiate Scan</span> to fire a
          synthetic acquisition through the trained 1D&nbsp;ResNet and read out
          a mineral class plus ilmenite mass fraction in milliseconds.
        </motion.p>

        <div className="flex flex-wrap gap-3 font-mono text-[10px] uppercase tracking-wider">
          <Pill label="schema" value={schemaVersion ?? "?"} ok={!!schemaVersion} />
          <Pill
            label="model"
            value={modelLoaded ? "ONNX online" : "offline"}
            ok={modelLoaded}
          />
          <Pill label="runtime" value="onnxruntime" ok />
        </div>
      </div>
    </header>
  );
}

function Pill({ label, value, ok }: { label: string; value: string; ok: boolean }) {
  const tone = ok
    ? "border-cyan-glow/30 text-cyan-glow shadow-glow-cyan"
    : "border-amber-glow/30 text-amber-glow shadow-glow-amber";
  return (
    <div
      className={`flex items-center gap-2 rounded-full border bg-void-800/60 px-3 py-1 ${tone}`}
    >
      <span className="text-slate-500">{label}</span>
      <span>{value}</span>
    </div>
  );
}
