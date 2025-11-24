"use client"

import { Moon, Sun } from "lucide-react"
import { useTheme } from "./theme-provider"

export function ThemeToggle() {
  const { theme, setTheme } = useTheme()

  return (
    <button
      className="btn-icon relative"
      onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
      aria-label="Toggle theme"
    >
      <Sun
        className={`h-5 w-5 transition-all ${theme === "dark" ? "opacity-0 scale-0 rotate-90 absolute" : "opacity-100 scale-100 rotate-0 relative"}`}
      />
      <Moon
        className={`h-5 w-5 transition-all ${theme === "light" ? "opacity-0 scale-0 -rotate-90 absolute" : "opacity-100 scale-100 rotate-0 relative"}`}
      />
    </button>
  )
}

