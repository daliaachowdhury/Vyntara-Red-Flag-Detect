/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        flag: {
          red: '#EF4444',
          yellow: '#F59E0B',
          green: '#10B981',
          neutral: '#94A3B8'
        }
      }
    },
  },
  plugins: [],
}

