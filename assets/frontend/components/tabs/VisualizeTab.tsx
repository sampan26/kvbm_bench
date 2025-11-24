import { KnowledgeGraphViewer } from "@/components/knowledge-graph-viewer"
import { Eye } from "lucide-react"

export function VisualizeTab() {
  return (
    <div className="nvidia-build-card p-0 overflow-hidden">
      <div className="p-6 border-b border-border/10">
        <div className="flex items-center gap-3 mb-3">
          <div className="w-8 h-8 rounded-lg bg-nvidia-green/15 flex items-center justify-center">
            <Eye className="h-4 w-4 text-nvidia-green" />
          </div>
          <h2 className="text-lg font-semibold text-foreground">Knowledge Graph Visualization</h2>
        </div>
        <p className="text-sm text-muted-foreground leading-relaxed">
          Explore connections between entities and visualize your knowledge graph
        </p>
      </div>
      
      <div className="p-6">
        <KnowledgeGraphViewer />
      </div>
    </div>
  )
} 