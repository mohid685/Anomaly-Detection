"use client";

import { useEffect, useState } from "react";
import UploadPanel from "@/components/UploadPanel";
import ResultPanel from "@/components/ResultPanel";
import CategorySelect from "@/components/CategorySelect";
import { fetchCategories, runPrediction, PredictionResult } from "@/lib/api";

export default function Home() {
  const [categories, setCategories] = useState<string[]>([]);
  const [selectedCategory, setSelectedCategory] = useState<string>("");
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchCategories()
      .then((cats) => {
        setCategories(cats);
        if (cats.length > 0) setSelectedCategory(cats[0]);
      })
      .catch(() => setError("Unable to reach the inspection backend."));
  }, []);

  async function handleInspect() {
    if (!file || !selectedCategory) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await runPrediction(selectedCategory, file);
      setResult(res);
    } catch (e: any) {
      setError(e.message || "Inspection failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="mx-auto max-w-6xl px-6 py-10">
      <div className="mb-8">
        <div className="text-xs uppercase tracking-widest text-neutral-500 mb-2">
          Component Category
        </div>
        <CategorySelect
          categories={categories}
          selected={selectedCategory}
          onSelect={setSelectedCategory}
        />
      </div>

      <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
        <div>
          <div className="text-xs uppercase tracking-widest text-neutral-500 mb-2">
            Input
          </div>
          <UploadPanel onFileSelected={setFile} disabled={loading} />
          <button
            onClick={handleInspect}
            disabled={!file || loading}
            className="mt-4 w-full border border-accent bg-accent/10 py-3 text-sm font-medium text-neutral-100 transition-colors hover:bg-accent/20 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {loading ? "Running Inspection..." : "Run Inspection"}
          </button>
          {error && (
            <div className="mt-3 border border-status-alert/40 bg-status-alert/5 p-3 text-xs text-status-alert font-mono">
              {error}
            </div>
          )}
        </div>

        <div>
          <div className="text-xs uppercase tracking-widest text-neutral-500 mb-2">
            Result
          </div>
          <ResultPanel result={result} />
        </div>
      </div>
    </main>
  );
}