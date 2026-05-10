import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}"
  ],
  theme: {
    extend: {
      colors: {
        bg:         "#0b0d12",
        panel:      "#11141c",
        panelHover: "#161923",
        text:       "#e8ecf3",
        muted:      "#8a95a3",
        accent:     "#7c9cff",
        accentSoft: "#2b3761",
        success:    "#34d399",
        warn:       "#fbbf24",
      },
      boxShadow: {
        soft:  "0 12px 40px -18px rgba(0,0,0,0.65)",
        glow:  "0 0 24px -6px rgba(124,156,255,0.35)",
      },
      borderColor: {
        DEFAULT: "rgba(255,255,255,0.08)",
      },
    }
  },
  plugins: []
};

export default config;
