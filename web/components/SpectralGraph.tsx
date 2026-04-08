"use client";

import { Activity } from "lucide-react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ReferenceLine, ResponsiveContainer,
} from "recharts";
import type { ScanState } from "@/lib/types";

interface Props {
  scanState: ScanState;
  spectralData: number[] | null;
  onScan: () => void;
}

const REF_LINES = [
  { x: 450, label: "Fe\u00B2\u207A", color: "#475569" },
  { x: 550, label: "Ol",             color: "#475569" },
  { x: 750, label: "Px",             color: "#475569" },
  { x: 850, label: "ILM",            color: "#0369a1" },
] as const;

function buildChartData(spec: number[]) {
  return spec.map((val, i) => ({
    nm: Math.round(340 + (i * 510) / 287),
    r: val,
  }));
}

export function SpectralGraph({ scanState, spectralData, onScan }: Props) {
  const data = spectralData ? buildChartData(spectralData) : null;
  const scanning = scanState === "scanning";

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="border-b border-slate-700 bg-slate-800 px-4 py-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Activity className="h-3.5 w-3.5 text-sky-400" />
            <span className="text-xs font-semibold uppercase tracking-widest text-slate-400">
              VIS/NIR SPECTRUM
            </span>
          </div>
          <span className="font-mono text-[10px] text-slate-500">
            340 &ndash; 850 nm &middot; 288 CH &middot; C12880MA
          </span>
        </div>
      </div>

      {/* Chart area */}
      <div className="flex-1 min-h-0 px-2 py-2">
        {data ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: 8 }}>
              <XAxis
                dataKey="nm"
                stroke="#334155"
                tick={{ fill: "#64748b", fontSize: 10, fontFamily: "ui-monospace" }}
                tickLine={false}
              />
              <YAxis
                stroke="transparent"
                tick={{ fill: "#64748b", fontSize: 10, fontFamily: "ui-monospace" }}
                tickFormatter={(v: number) => v.toFixed(2)}
                domain={[0, "dataMax + 0.1"]}
                tickLine={false}
              />
              {REF_LINES.map((rl) => (
                <ReferenceLine
                  key={rl.x}
                  x={rl.x}
                  stroke={rl.color}
                  strokeDasharray="3 4"
                  label={{ value: rl.label, fill: rl.color, fontSize: 9, position: "insideTopRight" }}
                />
              ))}
              <Tooltip
                contentStyle={{
                  backgroundColor: "#0f172a",
                  border: "1px solid #334155",
                  borderRadius: 0,
                  fontFamily: "ui-monospace",
                  fontSize: 11,
                  color: "#e2e8f0",
                }}
                formatter={(v: number) => [`I: ${v.toFixed(4)}`, ""]}
                labelFormatter={(v) => `\u03BB: ${v} nm`}
                separator=""
              />
              <Line
                type="monotone"
                dataKey="r"
                stroke="#38bdf8"
                strokeWidth={2}
                dot={false}
                animationDuration={1200}
                animationEasing="ease-in-out"
              />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="flex h-full items-center justify-center">
            <div className="flex flex-col items-center gap-2">
              <Activity className="h-8 w-8 text-slate-700" />
              <span className="font-mono text-xs uppercase tracking-widest text-slate-600">
                AWAITING C12880MA SYNC
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Scan button */}
      <div className="border-t border-slate-700 px-4 py-2">
        <button
          onClick={onScan}
          disabled={scanning}
          className={`w-full border py-2.5 font-mono text-xs font-semibold uppercase tracking-widest transition-colors ${
            scanning
              ? "border-amber-500 text-amber-400 bg-amber-500/5"
              : "border-sky-600 text-sky-400 hover:bg-sky-950"
          } disabled:cursor-not-allowed`}
        >
          {scanning ? "[ ACQUIRING... ]" : "[ INITIATE ACQUISITION SEQUENCE ]"}
        </button>
      </div>
    </div>
  );
}
