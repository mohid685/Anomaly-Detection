import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Industrial Defect Inspection System",
  description: "Unsupervised visual anomaly detection for industrial components.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">
        <div className="border-b border-surface-border bg-surface-panel/60 backdrop-blur">
          <div className="mx-auto max-w-6xl px-6 py-4 flex items-center justify-between">
            <div>
              <div className="text-sm tracking-widest text-neutral-400 uppercase">
                Automated Visual Inspection
              </div>
              <div className="text-lg font-semibold text-neutral-100">
                Defect Detection System
              </div>
            </div>
            <div className="text-xs text-neutral-500 font-mono">
              PatchCore &middot; Unsupervised Anomaly Detection
            </div>
          </div>
        </div>
        {children}
      </body>
    </html>
  );
}