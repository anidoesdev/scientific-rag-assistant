import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}"
  ],
  theme: {
    extend: {
      colors: {
        bg:         "#F7F4EF",
        panel:      "#EDE8DF",
        panelHover: "#E4DDD3",
        text:       "#1C2533",
        muted:      "#6B7A8D",
        accent:     "#B45309",
        accentSoft: "#FEF3C7",
        success:    "#059669",
        warn:       "#DC2626",
      },
      fontFamily: {
        sans:  ["var(--font-inter)", "ui-sans-serif", "system-ui"],
        serif: ["var(--font-lora)", "ui-serif", "Georgia"],
      },
      boxShadow: {
        soft:  "0 12px 40px -18px rgba(0,0,0,0.18)",
        glow:  "0 0 24px -6px rgba(180,83,9,0.22)",
      },
      borderColor: {
        DEFAULT: "rgba(0,0,0,0.09)",
      },
    }
  },
  plugins: []
};

export default config;
