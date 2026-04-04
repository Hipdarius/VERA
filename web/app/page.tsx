"use client";

import { useCallback, useEffect, useState } from "react";
import { motion } from "framer-motion";

import { Hero } from "@/components/Hero";
import { ScanButton } from "@/components/ScanButton";
import { MissionPanel } from "@/components/MissionPanel";
import { SpectrumChart } from "@/components/SpectrumChart";
import { ProbabilityBars } from "@/components/ProbabilityBars";
import { IlmeniteGauge } from "@/components/IlmeniteGauge";

import { fetchDemoPrediction, fetchMeta } from "@/lib/api";
import { CLASS_LABELS, type DemoResponse, type MetaResponse } from "@/lib/types";

export default function Home() {
  const [meta, setMeta] = useState<MetaResponse | null>(null);
  const [scan, setScan] = useState<DemoResponse | null>(null);
  const [isScanning, setIsScanning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Pull /api/meta once on mount so the hero pills can render schema
  // version + model status without blocking the user from scanning.
  useEffect(() => {
    fetchMeta()
      .then(setMeta)
      .catch((e) => setError(String(e)));
  }, []);

  const runScan = useCallback(async () => {
    setIsScanning(true);
    setError(null);
    const startedAt = performance.now();
    try {
      const result = await fetchDemoPrediction();
      // Hold the scanline animation visible for at least 700 ms — feels
      // less like an instant flicker on fast networks.
      const elapsed = performance.now() - startedAt;
      const remaining = Math.max(0, 700 - elapsed);
      if (remaining > 0) {
        await new Promise((r) => setTimeout(r, remaining));
      }
      setScan(result);
    } catch (e) {
      setError(String(e));
    } finally {
      setIsScanning(false);
    }
  }, []);

  return (
    <main className="relative min-h-screen pb-24">
      <Hero
        schemaVersion={meta?.schema_version ?? null}
        modelLoaded={meta?.model_loaded ?? false}
      />

      <div className="mx-auto mt-10 flex max-w-6xl flex-col gap-6 px-6">
        {/* Action bar */}
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="flex flex-wrap items-center justify-between gap-4 rounded-xl border border-cyan-glow/15 bg-void-800/70 px-5 py-4 backdrop-blur-sm"
        >
          <div className="flex flex-col gap-1">
            <span className="font-mono text-[10px] uppercase tracking-widest text-slate-500">
              Probe operation
            </span>
            <span className="font-mono text-sm text-slate-200">
              Synthetic acquisition · 288 ch · 340–850 nm
            </span>
          </div>
          <ScanButton
            onClick={runScan}
            isScanning={isScanning}
            disabled={!meta?.model_loaded}
          />
        </motion.div>

        {error && (
          <div className="rounded-lg border border-amber-glow/30 bg-amber-glow/10 px-4 py-3 font-mono text-xs text-amber-glow">
            {error}
          </div>
        )}

        {/* Top row: spectrum + classification */}
        <div className="grid gap-6 lg:grid-cols-3">
          <div className="lg:col-span-2">
            <MissionPanel title="VIS/NIR Spectrum">
              <SpectrumChart
                wavelengths={meta?.wavelengths_nm ?? null}
                spectrum={scan?.spec ?? null}
              />
              <div className="mt-3 flex flex-wrap gap-4 font-mono text-[10px] uppercase tracking-widest text-slate-500">
                <span>340 nm – 850 nm</span>
                <span>·</span>
                <span>Hamamatsu C12880MA emulation</span>
                {scan && (
                  <>
                    <span>·</span>
                    <span className="text-cyan-glow">
                      LIF 450lp = {scan.lif_450lp.toFixed(3)}
                    </span>
                  </>
                )}
              </div>
            </MissionPanel>
          </div>

          <MissionPanel title="Ilmenite Regression" delay={0.05}>
            <IlmeniteGauge
              fraction={scan?.ilmenite_fraction ?? null}
              trueFraction={scan?.true_ilmenite_fraction ?? null}
            />
          </MissionPanel>
        </div>

        {/* Bottom row: classification + telemetry */}
        <div className="grid gap-6 lg:grid-cols-3">
          <MissionPanel title="Mineral Class Posterior" delay={0.1}>
            <ProbabilityBars
              probabilities={scan?.probabilities ?? null}
              predictedClass={scan?.predicted_class ?? null}
            />
          </MissionPanel>

          <MissionPanel title="Mission Telemetry" delay={0.15}>
            <dl className="space-y-4">
              <Telemetry
                label="Predicted class"
                value={
                  scan
                    ? CLASS_LABELS[scan.predicted_class] ?? scan.predicted_class
                    : "—"
                }
                glow="cyan"
              />
              <Telemetry
                label="Confidence"
                value={scan ? `${(scan.confidence * 100).toFixed(1)}%` : "—"}
                glow="cyan"
              />
              <Telemetry
                label="Ground truth"
                value={scan ? CLASS_LABELS[scan.true_class] ?? scan.true_class : "—"}
                glow="amber"
              />
              <Telemetry
                label="Match"
                value={
                  scan
                    ? scan.predicted_class === scan.true_class
                      ? "NOMINAL"
                      : "DEVIATION"
                    : "—"
                }
                glow={
                  scan
                    ? scan.predicted_class === scan.true_class
                      ? "cyan"
                      : "amber"
                    : "cyan"
                }
              />
            </dl>
          </MissionPanel>

          <MissionPanel title="Model Provenance" delay={0.2}>
            <dl className="space-y-4">
              <Telemetry
                label="Model"
                value={scan?.model_version ?? "regoscan-resnet"}
                glow="cyan"
              />
              <Telemetry
                label="Schema"
                value={meta?.schema_version ?? "—"}
                glow="cyan"
              />
              <Telemetry
                label="ONNX SHA-256"
                value={meta?.model_sha256 ?? "—"}
                glow="cyan"
                mono
              />
              <Telemetry
                label="Features"
                value={meta ? `${meta.n_features_total}` : "—"}
                glow="cyan"
              />
            </dl>
          </MissionPanel>
        </div>

        <footer className="mt-6 flex flex-col items-center gap-1 text-center font-mono text-[10px] uppercase tracking-widest text-slate-600">
          <span>Regoscan · 1D ResNet · ONNXRuntime · FastAPI</span>
          <span>
            Inference at the edge — same model that fits the embedded probe.
          </span>
        </footer>
      </div>
    </main>
  );
}

function Telemetry({
  label,
  value,
  glow,
  mono = false,
}: {
  label: string;
  value: string;
  glow: "cyan" | "amber";
  mono?: boolean;
}) {
  const tone = glow === "cyan" ? "text-cyan-glow" : "text-amber-glow";
  return (
    <div className="flex items-center justify-between gap-4 border-b border-void-700 pb-2 last:border-b-0 last:pb-0">
      <dt className="font-mono text-[10px] uppercase tracking-widest text-slate-500">
        {label}
      </dt>
      <dd className={`font-mono ${mono ? "text-[11px]" : "text-sm"} ${tone}`}>
        {value}
      </dd>
    </div>
  );
}
