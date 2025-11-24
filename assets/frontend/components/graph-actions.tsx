"use client"

import { Network, Zap, HelpCircle } from "lucide-react"
import { useDocuments } from "@/contexts/document-context"
import { Loader2 } from "lucide-react"
import { useState } from "react"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"

export function GraphActions() {
  const { documents, processDocuments, isProcessing, openGraphVisualization } = useDocuments()
  const [useLangChain, setUseLangChain] = useState(false)

  const hasNewDocuments = documents.some((doc) => doc.status === "New")
  const hasProcessedDocuments = documents.some(
    (doc) => doc.status === "Processed" && doc.triples && doc.triples.length > 0,
  )

  const handleProcessDocuments = async () => {
    try {
      // Get IDs of documents with "New" status
      const newDocumentIds = documents
        .filter(doc => doc.status === "New")
        .map(doc => doc.id);
        
      if (newDocumentIds.length === 0) {
        console.log("No new documents to process");
        return;
      }
      
      await processDocuments(newDocumentIds, {
        useLangChain,
        useGraphTransformer: false,
        promptConfigs: undefined
      });
    } catch (error) {
      console.error('Error processing documents:', error);
    }
  }

  return (
    <div className="flex gap-3 items-center">
      <div className="flex items-center space-x-2 mr-2">
        <div className="flex items-center gap-1.5">
          <Switch
            id="use-langchain-graph"
            checked={useLangChain}
            onCheckedChange={(value) => {
              setUseLangChain(value);
              // Dispatch custom event to update other components
              window.dispatchEvent(new CustomEvent('langChainToggled', { 
                detail: { useLangChain: value } 
              }));
            }}
          />
          <Label htmlFor="use-langchain-graph" className="text-xs font-medium">Use LangChain</Label>
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <HelpCircle className="h-3.5 w-3.5 text-muted-foreground cursor-help" />
              </TooltipTrigger>
              <TooltipContent className="max-w-[220px] p-3">
                <p className="text-xs">
                  Enabling LangChain uses AI-powered knowledge extraction for more accurate triple generation.
                </p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      </div>
      <button
        className={`btn-primary ${!hasNewDocuments || isProcessing ? "opacity-60 cursor-not-allowed" : ""}`}
        disabled={!hasNewDocuments || isProcessing}
        onClick={handleProcessDocuments}
      >
        {isProcessing ? (
          <>
            <Loader2 className="h-4 w-4 animate-spin" />
            Processing...
          </>
        ) : (
          <>
            <Zap className="h-4 w-4" />
            Process Documents
          </>
        )}
      </button>
      <button
        className={`btn-primary ${!hasProcessedDocuments || isProcessing ? "opacity-60 cursor-not-allowed" : ""}`}
        disabled={!hasProcessedDocuments || isProcessing}
        onClick={() => openGraphVisualization()}
      >
        <Network className="h-4 w-4" />
        View Knowledge Graph
      </button>
    </div>
  )
}

