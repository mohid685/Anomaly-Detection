import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        surface: {
          DEFAULT: "#0d0f12",
          panel: "#14171b",
          border: "#22262c",
        },
        accent: {
          DEFAULT: "#5b8bd6",
          muted: "#3d5a85",
        },
        status: {
          ok: "#4a9d6e",
          alert: "#b5544a",
        },
      },
      fontFamily: {
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
    },
  },
  plugins: [],
};
export default config;