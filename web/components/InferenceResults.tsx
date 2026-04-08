"use client";

import { Crosshair } from "lucide-react";
import type { DemoResponse, ScanHistoryEntry, ScanState } from "@/lib/types";
import { CLASS_LABELS } from "@/lib/types";

interface Props {
  scanState: ScanState;
  result: DemoResponse | null;
  scanHistory: ScanHistoryEntry[];
}

const CLASSES = ["ilmenite_rich", "olivine_rich", "pyroxene_rich", "anorthositic", "mixed"];
const TICKS = [0, 25, 50, 75, 100];

function confidenceColor(c: number): string {
  if (c >= 0.85) return "text-emerald-400";
  if (c >= 0.60) return "text-amber-400";
  return "text-rose-500";
}

export function InferenceResults({ scanState, result, scanHistory }: Props) {
  const ilmPct = result ? result.ilmenite_fraction * 100 : 0;
  const probMap = result
    ? Object.fromEntries(result.probabilities.map((p) => [p.name, p.probability]))
    : null;

  return (
    <div className="flex h-full flex-col">
      <div className="border-b border-slate-700 bg-slate-800 px-4 py-2">
        <div className="flex items-center gap-2">
          <Crosshair className="h-3.5 w-3.5 text-sky-400" />
          <span className="text-xs font-semibold uppercase tracking-widest text-slate-400">
            INFERENCE RESULTS
          </span>
        </div>
      </div>
      <div className="flex-1 overflow-y-auto px-4 py-3 space-y-4">
        {/* Mineral class readout */}
        <div>
          <span className="text-xs font-semibold uppercase tracking-widest text-slate-400">
            MINERAL CLASS
          </span>
          <div className="mt-1 border border-slate-800 bg-slate-950 px-3 py-2 flex items-center justify-between">
            <span className={`font-mono text-sm ${result ? confidenceColor(result.confidence) : "text-slate-600"}`}>
              {result ? (CLASS_LABELS[result.predicted_class] ?? result.predicted_class).toUpperCase() : "BUFFER EMPTY"}
            </span>
            {result && (
              <span className={`font-mono text-xs ${confidenceColor(result.confidence)}`}>
                {(result.confidence * 100).toFixed(1)}%
              </span>
            )}
          </div>
        </div>

        {/* Class posterior bars */}
        <div>
          <span className="text-xs font-semibold uppercase tracking-widest text-slate-400">
            CLASS POSTERIOR
          </span>
          <div className="mt-2 space-y-2">
            {CLASSES.map((cls) => {
              const p = probMap?.[cls] ?? 0;
              const isTop = result?.predicted_class === cls;
              return (
                <div key={cls}>
                  <div className="flex items-center justify-between mb-0.5">
                    <span className={`font-mono text-[10px] ${isTop ? "text-sky-400" : "text-slate-500"}`}>
                      {(CLASS_LABELS[cls] ?? cls).toUpperCase()}
                    </span>
                    <span className={`font-mono text-[10px] ${isTop ? "text-sky-400" : "text-slate-600"}`}>
                      {result ? `${(p * 100).toFixed(1)}%` : "--.-%"}
                    </span>
                  </div>
                  <div className="h-1 w-full bg-slate-800">
                    {result && <div className="h-full bg-sky-500 transition-all duration-700" style={{ width: `${p * 100}%` }} />}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Ilmenite gauge */}
        <div className="border-t border-slate-800 pt-3">
          <span className="text-xs font-semibold uppercase tracking-widest text-slate-400">
            ILMENITE MASS FRACTION
          </span>
          <div className="flex items-baseline gap-1 mt-1">
            <span className="font-mono text-2xl text-sky-400">
              {result ? ilmPct.toFixed(1) : "--.-"}
            </span>
            <span className="font-mono text-xs text-slate-500">%</span>
          </div>
          <div className="relative mt-2 h-2.5 w-full bg-slate-800">
            <div
              className="absolute inset-y-0 left-0 bg-sky-500 transition-all duration-700"
              style={{ width: `${result ? Math.min(ilmPct, 100) : 0}%` }}
            />
            {TICKS.map((t) => (
              <div key={t} className="absolute top-0 h-full w-px bg-slate-600" style={{ left: `${t}%` }} />
            ))}
          </div>
          <div className="flex justify-between mt-0.5">
            {TICKS.map((t) => (
              <span key={t} className="font-mono text-[8px] text-slate-600">{t}</span>
            ))}
          </div>
        </div>

        {/* Model provenance */}
        <div className="border-t border-slate-800 pt-3 space-y-1">
          <span className="text-xs font-semibold uppercase tracking-widest text-slate-400">MODEL</span>
          <MetaRow label="ENGINE" value="ONNXRUNTIME" accent />
          <MetaRow label="ARCH" value="1D-RESNET" />
          <MetaRow label="FEATURES" value="301" />
        </div>

        {/* Scan history */}
        <div className="border-t border-slate-800 pt-3">
          <span className="text-xs font-semibold uppercase tracking-widest text-slate-400">SCAN HISTORY</span>
          {scanHistory.length === 0 ? (
            <p className="mt-1 font-mono text-[10px] text-slate-600">NO PRIOR ACQUISITIONS</p>
          ) : (
            <div className="mt-1 space-y-1">
              {scanHistory.map((e, i) => (
                <div key={i} className="flex items-center justify-between">
                  <span className="font-mono text-[10px] text-slate-500">
                    {(CLASS_LABELS[e.predicted_class] ?? e.predicted_class).toUpperCase()}
                  </span>
                  <span className={`font-mono text-[10px] ${confidenceColor(e.confidence)}`}>
                    ILM {(e.ilmenite_fraction * 100).toFixed(1)}% · {(e.confidence * 100).toFixed(0)}%
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function MetaRow({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className="flex items-center justify-between">
      <span className="font-mono text-[10px] text-slate-500">{label}</span>
      <span className={`font-mono text-[10px] ${accent ? "text-emerald-400" : "text-slate-400"}`}>{value}</span>
    </div>
  );
}
