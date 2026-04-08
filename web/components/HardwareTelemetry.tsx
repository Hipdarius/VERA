"use client";

import { useEffect, useState } from "react";
import { Cpu, Thermometer, Clock, Zap } from "lucide-react";
import type { ScanState } from "@/lib/types";

const LED_CONFIG = [
  { nm: "385", border: "border-l-violet-400" },
  { nm: "405", border: "border-l-violet-300" },
  { nm: "450", border: "border-l-blue-500" },
  { nm: "500", border: "border-l-green-400" },
  { nm: "525", border: "border-l-green-500" },
  { nm: "590", border: "border-l-yellow-400" },
  { nm: "625", border: "border-l-orange-400" },
  { nm: "660", border: "border-l-orange-500" },
  { nm: "730", border: "border-l-red-500" },
  { nm: "780", border: "border-l-red-600" },
  { nm: "850", border: "border-l-red-700" },
  { nm: "940", border: "border-l-red-900" },
] as const;

interface Props {
  scanState: ScanState;
  ledValues: number[] | null;
  lifValue: number | null;
}

function useJitter(min: number, max: number, intervalMs: number): number {
  const [val, setVal] = useState(min + (max - min) / 2);
  useEffect(() => {
    const id = setInterval(() => {
      setVal(min + Math.random() * (max - min));
    }, intervalMs);
    return () => clearInterval(id);
  }, [min, max, intervalMs]);
  return val;
}

export function HardwareTelemetry({ scanState, ledValues, lifValue }: Props) {
  const intTime = useJitter(9.8, 10.2, 3000);
  const temp = useJitter(22.1, 22.4, 3000);

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b border-slate-800 bg-slate-900 px-4 py-2">
        <span className="text-xs font-semibold uppercase tracking-widest text-slate-400">
          HARDWARE TELEMETRY
        </span>
        <span className="font-mono text-[10px] text-emerald-400 animate-blink">
          NOMINAL
        </span>
      </div>
      <div className="flex-1 overflow-y-auto px-4 py-3 space-y-2">
        <Row icon={Cpu} label="MCU" value="ESP32-S3" />
        <Row icon={Zap} label="SENSOR" value="C12880MA" />
        <Row icon={Clock} label="INT TIME" value={`${intTime.toFixed(1)} ms`} />
        <Row icon={Thermometer} label="TEMP" value={`${temp.toFixed(1)} C`} />

        <div className="mt-3 border-t border-slate-800 pt-3">
          <span className="text-xs font-semibold uppercase tracking-widest text-slate-400">
            LED ARRAY
          </span>
          <div className="mt-2 space-y-0.5">
            {LED_CONFIG.map((led, i) => (
              <div
                key={led.nm}
                className={`flex items-center justify-between border-l-2 ${led.border} py-0.5 pl-2`}
              >
                <span className="font-mono text-[10px] text-slate-500">
                  {led.nm} nm
                </span>
                <div className="flex items-center gap-2">
                  {ledValues ? (
                    <span className="font-mono text-[10px] text-emerald-400">
                      {ledValues[i].toFixed(3)}
                    </span>
                  ) : null}
                  <span
                    className={`h-1.5 w-1.5 ${
                      scanState === "scanning" ? "bg-amber-500" : "bg-slate-700"
                    }`}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="mt-3 border-t border-slate-800 pt-3">
          <span className="text-xs font-semibold uppercase tracking-widest text-slate-400">
            LIF CHANNEL
          </span>
          <div className="mt-2 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span
                className={`h-2 w-2 rounded-full bg-violet-500 ${
                  scanState === "scanning" ? "animate-blink" : ""
                }`}
              />
              <span className="font-mono text-[10px] text-slate-500">450 LP</span>
            </div>
            <span className="font-mono text-[10px] text-emerald-400">
              {lifValue !== null ? lifValue.toFixed(4) : "STANDBY"}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

function Row({
  icon: Icon,
  label,
  value,
}: {
  label: string;
  value: string;
  icon: React.ComponentType<{ className?: string }>;
}) {
  return (
    <div className="flex items-center justify-between border-b border-slate-800 py-1.5 last:border-b-0">
      <div className="flex items-center gap-2">
        <Icon className="h-3.5 w-3.5 text-slate-500" />
        <span className="text-xs font-semibold uppercase tracking-widest text-slate-400">{label}</span>
      </div>
      <span className="font-mono text-sm text-emerald-400">{value}</span>
    </div>
  );
}
