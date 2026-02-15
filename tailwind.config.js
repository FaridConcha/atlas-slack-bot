/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./app/**/*.{js,ts,jsx,tsx}", "./components/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        slateBrand: "#22282b",
        goldBrand: "#f2cc57",
        lightGrayBrand: "#e6e6ed",
        darkGrayBrand: "#697684",
        skyBlueBrand: "#8fcae9",
        greenBrand: "#56b784"
      },
      fontFamily: {
        heading: ["Exo 2", "sans-serif"],
        body: ["Nunito", "sans-serif"]
      }
    }
  },
  plugins: []
};
