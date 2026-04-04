import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Regoscan Mission Console",
  description:
    "VIS/NIR + 405 nm LIF probe for in-situ lunar regolith mineral classification",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-void-900 text-slate-200 antialiased">
        <div className="relative z-10">{children}</div>
      </body>
    </html>
  );
}
