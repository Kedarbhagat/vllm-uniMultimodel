// tailwind.config.js
/** @type {import('tailwindcss').Config} */
module.exports = {
  // Add this line to enable dark mode based on the 'dark' class
  darkMode: 'class', 
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}