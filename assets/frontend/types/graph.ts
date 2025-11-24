/**
 * Triple interface representing a knowledge graph edge
 */
export interface Triple {
  subject: string
  predicate: string
  object: string
  confidence?: number
  usedFallback?: boolean
}

// Add this interface to the file
export interface VectorDBStats {
  nodes: number;
  relationships: number;
  source: string;
  httpHealthy?: boolean;
} 