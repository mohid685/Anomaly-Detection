"use client";

import { useRef, useState } from "react";

interface Props {
  onFileSelected: (file: File) => void;
  disabled?: boolean;
}

export default function UploadPanel({ onFileSelected, disabled }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);

  function handleFile(file: File) {
    if (!file.type.startsWith("image/")) return;
    setPreviewUrl(URL.createObjectURL(file));
    onFileSelected(file);
  }

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
      onDragLeave={() => setDragActive(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragActive(false);
        const file = e.dataTransfer.files?.[0];
        if (file) handleFile(file);
      }}
      onClick={() => !disabled && inputRef.current?.click()}
      className={`scan-grid relative flex h-72 cursor-pointer flex-col items-center justify-center border transition-colors ${
        dragActive ? "border-accent bg-accent/5" : "border-surface-border bg-surface-panel"
      } ${disabled ? "opacity-50 cursor-not-allowed" : ""}`}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        disabled={disabled}
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) handleFile(file);
        }}
      />

      {previewUrl ? (
        <img src={previewUrl} alt="Selected component" className="max-h-64 max-w-full object-contain" />
      ) : (
        <div className="text-center">
          <div className="text-sm font-medium text-neutral-400">
            Drop component image here or click to browse
          </div>
          <div className="mt-1 text-xs text-neutral-600 font-mono">
            PNG / JPG &middot; single component per frame
          </div>
        </div>
      )}
    </div>
  );
}