"use client"

import dynamic from "next/dynamic"
import type { Triple } from "@/utils/text-processing"

// Dynamically import the GraphVisualization component with no SSR
// This allows for GPU-accelerated WebGL rendering
const DynamicGraphVisualization = dynamic(
  () => import("./graph-visualization").then((mod) => ({ default: mod.GraphVisualization })),
  {
    ssr: false,
    loading: () => (
      <div className="h-[400px] bg-gray-900 rounded-lg flex items-center justify-center">
        <div className="text-nvidia-green">Loading GPU-accelerated graph visualization...</div>
      </div>
    ),
  },
)

interface DynamicGraphProps {
  triples: Triple[]
}

export function DynamicGraph({ triples }: DynamicGraphProps) {
  return <DynamicGraphVisualization triples={triples} />
}

