import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}"
  ],
  theme: {
    extend: {
      colors: {
        bg: "#0b0d12",
        panel: "#11141c",
        text: "#e8ecf3",
        muted: "#9aa4b2",
        accent: "#7c9cff",
        accentSoft: "#2b3761"
      },
      boxShadow: {
        soft: "0 12px 40px -18px rgba(0,0,0,0.6)"
      }
    }
  },
  plugins: []
};

export default config;
