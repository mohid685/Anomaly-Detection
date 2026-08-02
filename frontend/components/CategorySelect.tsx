"use client";

interface Props {
  categories: string[];
  selected: string;
  onSelect: (category: string) => void;
}

const LABELS: Record<string, string> = {
  metal_nut: "Metal Nut",
  screw: "Screw",
  cable: "Cable",
  transistor: "Transistor",
};

export default function CategorySelect({ categories, selected, onSelect }: Props) {
  return (
    <div className="flex flex-wrap gap-2">
      {categories.map((cat) => (
        <button
          key={cat}
          onClick={() => onSelect(cat)}
          className={`px-4 py-2 text-sm font-medium border transition-colors ${
            selected === cat
              ? "border-accent bg-accent/10 text-neutral-100"
              : "border-surface-border bg-surface-panel text-neutral-400 hover:border-neutral-600 hover:text-neutral-200"
          }`}
        >
          {LABELS[cat] || cat}
        </button>
      ))}
    </div>
  );
}