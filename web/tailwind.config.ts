import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        // "Modern Space Mission" palette — deep void backgrounds with
        // cyan instrumentation accents and amber alert highlights.
        void: {
          900: "#05060a",
          800: "#0a0d14",
          700: "#10141d",
          600: "#161b27",
          500: "#1f2533",
        },
        cyan: {
          glow: "#22d3ee",
          dim: "#0e7490",
        },
        amber: {
          glow: "#fbbf24",
          dim: "#b45309",
        },
      },
      fontFamily: {
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      boxShadow: {
        "glow-cyan": "0 0 24px rgba(34, 211, 238, 0.35)",
        "glow-amber": "0 0 24px rgba(251, 191, 36, 0.35)",
      },
      animation: {
        "scan-line": "scanLine 2.4s ease-in-out infinite",
        "pulse-soft": "pulseSoft 3s ease-in-out infinite",
      },
      keyframes: {
        scanLine: {
          "0%": { transform: "translateY(0%)", opacity: "0" },
          "10%": { opacity: "1" },
          "90%": { opacity: "1" },
          "100%": { transform: "translateY(100%)", opacity: "0" },
        },
        pulseSoft: {
          "0%, 100%": { opacity: "0.4" },
          "50%": { opacity: "1" },
        },
      },
    },
  },
  plugins: [],
};

export default config;
