/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      colors: {
        border: "hsl(240 5% 20%)",
        background: "hsl(240 10% 6%)",
        foreground: "hsl(0 0% 98%)",
        muted: "hsl(240 5% 15%)",
        "muted-foreground": "hsl(240 5% 65%)",
        primary: "hsl(217 91% 60%)",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
};
