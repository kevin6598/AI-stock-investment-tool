/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        up: "#34d399",      // emerald-400
        down: "#f43f5e",    // rose-500
        accent: "#22d3ee",  // cyan-400
        surface: {
          DEFAULT: "#111827",  // gray-900
          light: "#ffffff",
        },
        card: {
          DEFAULT: "#1f2937",  // gray-800
          light: "#f9fafb",    // gray-50
        },
      },
    },
  },
  plugins: [],
};
