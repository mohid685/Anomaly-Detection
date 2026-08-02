const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

export interface PredictionResult {
  category: string;
  score: number;
  threshold: number;
  is_defective: boolean;
  heatmap_base64: string;
}

export async function fetchCategories(): Promise<string[]> {
  const res = await fetch(`${API_BASE}/api/categories`);
  if (!res.ok) throw new Error("Failed to fetch categories");
  const data = await res.json();
  return data.categories;
}

export async function runPrediction(category: string, file: File): Promise<PredictionResult> {
  const formData = new FormData();
  formData.append("category", category);
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/api/predict`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Prediction request failed." }));
    throw new Error(err.detail || "Prediction request failed.");
  }

  return res.json();
}