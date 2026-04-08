"use client";

import { useEffect, useState } from "react";
import { Cpu, Wifi } from "lucide-react";

function useUtcClock(): string {
  const [time, setTime] = useState("");
  useEffect(() => {
    setTime(formatUtc(new Date()));
    const id = setInterval(() => setTime(formatUtc(new Date())), 1000);
    return () => clearInterval(id);
  }, []);
  return time;
}

function formatUtc(d: Date): string {
  return d.toISOString().slice(11, 19);
}

export function TopNav() {
  const utc = useUtcClock();
  const [signalOn, setSignalOn] = useState(true);

  useEffect(() => {
    const id = setInterval(() => setSignalOn((v) => !v), 2000);
    return () => clearInterval(id);
  }, []);

  return (
    <>
      {/* 1px sky-500 "power on" indicator strip */}
      <div className="h-px bg-sky-500" />
      <nav className="relative flex items-center justify-between border-b border-slate-800 bg-slate-900 px-4 py-2">
        <div className="flex items-center gap-3">
          <Cpu className="h-4 w-4 text-sky-400" />
          <span className="text-xs font-semibold uppercase tracking-widest text-slate-300">
            VERA // VISIBLE & EMISSION REGOLITH ASSESSMENT
          </span>
        </div>

        {/* Center: schema/model version tag */}
        <div className="absolute inset-x-0 flex justify-center pointer-events-none">
          <span className="font-mono text-[10px] text-slate-500">
            SCHEMA v1.0.0 · CNN_V2 · ONNX
          </span>
        </div>

        <div className="flex items-center gap-4">
          <span className="font-mono text-xs text-slate-400" suppressHydrationWarning>
            UTC {utc}
          </span>
          <div className="flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-emerald-400 animate-blink" />
            <Wifi
              className={`h-3.5 w-3.5 transition-colors duration-700 ${
                signalOn ? "text-emerald-400" : "text-slate-500"
              }`}
            />
            <span className="text-xs font-semibold uppercase tracking-widest text-emerald-400">
              LINK ESTABLISHED
            </span>
          </div>
        </div>
      </nav>
    </>
  );
}
