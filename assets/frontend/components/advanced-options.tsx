import React, { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";

interface AdvancedOptionsProps {
  title?: string;
  children: React.ReactNode;
  className?: string;
  defaultOpen?: boolean;
}

export function AdvancedOptions({
  title = "Advanced Options",
  children,
  className,
  defaultOpen = false
}: AdvancedOptionsProps) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div className={cn("border rounded-md overflow-hidden", className)}>
      <div 
        className="flex items-center justify-between p-3 bg-muted/30 cursor-pointer hover:bg-muted/50 transition-colors"
        onClick={() => setIsOpen(!isOpen)}
      >
        <h3 className="text-sm font-medium flex items-center">
          {isOpen ? (
            <ChevronDown className="h-4 w-4 mr-2" />
          ) : (
            <ChevronRight className="h-4 w-4 mr-2" />
          )}
          {title}
        </h3>
      </div>
      
      {isOpen && (
        <div className="p-4 border-t border-border/50">
          {children}
        </div>
      )}
    </div>
  );
} 