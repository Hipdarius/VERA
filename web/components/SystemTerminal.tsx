"use client";

import { useEffect, useRef } from "react";
import { Terminal } from "lucide-react";
import type { TerminalLine } from "@/lib/types";

const BOOT_LOG: TerminalLine[] = [
  { timestamp: "[00:00.000]", text: "VERA v1.0 BOOT SEQUENCE INITIATED", level: "info" },
  { timestamp: "[00:00.012]", text: "schema.py loaded \u2014 SCHEMA_VERSION 1.0.0", level: "info" },
  { timestamp: "[00:00.034]", text: "feature vector contract: 288 spec + 12 led + 1 lif = 301", level: "info" },
  { timestamp: "[00:00.051]", text: "ONNX inference engine initialized \u2014 CPUExecutionProvider", level: "info" },
  { timestamp: "[00:00.089]", text: "model.onnx loaded \u2014 1D ResNet (673,254 params)", level: "success" },
  { timestamp: "[00:00.102]", text: "endpoints ready: /api/predict, /api/predict/demo, /api/meta", level: "info" },
  { timestamp: "[00:00.115]", text: "LINK ESTABLISHED \u2014 awaiting sensor frames", level: "success" },
];

const MAX_LINES = 40;

const LINE_COLORS: Record<TerminalLine["level"], string> = {
  info: "text-slate-500",
  success: "text-emerald-400",
  warn: "text-amber-400",
  error: "text-rose-500",
};

interface Props {
  lines: TerminalLine[];
}

export function SystemTerminal({ lines }: Props) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const allLines = [...BOOT_LOG, ...lines].slice(-MAX_LINES);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [lines.length]);

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-2 border-b border-slate-700 bg-slate-800 px-4 py-1.5">
        <Terminal className="h-3.5 w-3.5 text-sky-400" />
        <span className="text-xs font-semibold uppercase tracking-widest text-slate-400">
          SYSTEM LOG
        </span>
        <span className="ml-auto font-mono text-[9px] text-slate-600">
          {allLines.length} / {MAX_LINES}
        </span>
      </div>
      <div ref={scrollRef} className="flex-1 overflow-y-auto bg-slate-950 px-4 py-2">
        {allLines.map((line, i) => (
          <div key={i} className={`font-mono text-[11px] leading-relaxed ${LINE_COLORS[line.level]}`}>
            <span className="text-slate-600">{line.timestamp}</span> {line.text}
          </div>
        ))}
        <div className="mt-0.5 font-mono text-[11px] text-emerald-400">
          &gt; _
        </div>
      </div>
    </div>
  );
}
