import React from "react"
import { Loader2, LucideProps } from "lucide-react"
import { cn } from "@/lib/utils"

interface SpinnerProps extends Omit<LucideProps, "ref"> {
  size?: "sm" | "md" | "lg"
}

export function Spinner({ size = "md", className, ...props }: SpinnerProps) {
  const sizeClasses = {
    sm: "h-3 w-3",
    md: "h-4 w-4",
    lg: "h-6 w-6",
  }

  return (
    <Loader2 
      className={cn(`animate-spin ${sizeClasses[size]}`, className)} 
      {...props} 
    />
  )
} 