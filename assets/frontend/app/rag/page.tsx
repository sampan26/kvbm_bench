"use client";

import { useState, useEffect } from "react";
import { RagQuery, RagParams } from "@/components/rag-query";
import type { Triple } from "@/types/graph";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { DatabaseConnection } from "@/components/database-connection";
import { NvidiaIcon } from "@/components/nvidia-icon";
import { ArrowLeft, BarChart2, Search as SearchIcon } from "lucide-react";

export default function RagPage() {
  const router = useRouter();
  const [results, setResults] = useState<Triple[] | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [vectorEnabled, setVectorEnabled] = useState(false);
  const [metrics, setMetrics] = useState<{
    avgQueryTime: number;
    avgRelevance: number;
    precision: number;
    recall: number;
  } | null>(null);
  const [currentParams, setCurrentParams] = useState<RagParams>({
    kNeighbors: 4096,
    fanout: 400,
    numHops: 2,
    topK: 5,
    useVectorSearch: false,
    usePureRag: false,
    queryMode: 'traditional'
  });

  // Initialize backend when the page loads
  useEffect(() => {
    // Initialize the backend services
    const initializeBackend = async () => {
      try {
        // Check graph database connection (ArangoDB by default)
        const graphResponse = await fetch('/api/graph-db', {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
        });
        
        if (!graphResponse.ok) {
          const errorData = await graphResponse.json();
          console.warn('Graph database connection warning:', errorData.error);
        }
        
        // Check if vector search is available
        const vectorResponse = await fetch('/api/pinecone-diag/stats');
        if (vectorResponse.ok) {
          const data = await vectorResponse.json();
          setVectorEnabled(data.totalVectorCount > 0);
        }
        
        // Fetch basic metrics
        const metricsResponse = await fetch('/api/metrics');
        if (metricsResponse.ok) {
          const data = await metricsResponse.json();
          setMetrics({
            avgQueryTime: data.avgQueryTime,
            avgRelevance: data.avgRelevance,
            precision: data.precision,
            recall: data.recall
          });
        }
      } catch (error) {
        console.warn('Error initializing backends:', error);
      }
    };

    initializeBackend();
  }, []);

  const handleQuerySubmit = async (query: string, params: RagParams) => {
    setIsLoading(true);
    setErrorMessage(null);
    setCurrentParams(params); // Store current params for UI rendering
    const startTime = Date.now();
    let queryMode: 'pure-rag' | 'vector-search' | 'traditional' = 'traditional';
    let resultCount = 0;
    let relevanceScore = 0;
    
    try {
      // If using pure RAG (Pinecone + LangChain) without graph search
      if (params.usePureRag) {
        queryMode = 'pure-rag';
        try {
          console.log('Using pure RAG with just Pinecone and LangChain for query:', query);
          const ragResponse = await fetch('/api/rag-query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              query,
              topK: params.topK
            })
          });
          
          if (ragResponse.ok) {
            const data = await ragResponse.json();
            // Handle the answer - we might need to display differently than triples
            if (data.answer) {
              // Special UI handling for text answer rather than triples
              setResults([{
                subject: 'Answer',
                predicate: '',
                object: data.answer,
                usedFallback: data.usedFallback
              }]);
              
              resultCount = 1;
              relevanceScore = data.relevanceScore || 0;
              
              // Log the query with performance metrics
              logQuery(query, queryMode, {
                executionTimeMs: Date.now() - startTime,
                relevanceScore,
                resultCount
              });
              
              console.log('Pure RAG query completed successfully');
              setIsLoading(false);
              return;
            }
          } else {
            // If the RAG query fails, log but continue to try other methods
            const errorData = await ragResponse.json();
            throw new Error(errorData.error || 'Failed to execute pure RAG query');
          }
        } catch (ragError) {
          console.warn('Pure RAG query error (falling back to other methods):', ragError);
          // Continue to other query methods as fallback
        }
      }
      
      // If we have vector embeddings, use enhanced query with metadata
      if (vectorEnabled && params.useVectorSearch) {
        queryMode = 'vector-search';
        try {
          console.log('Using enhanced RAG with LangChain for query:', query);
          const enhancedResponse = await fetch('/api/enhanced-query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              query,
              kNeighbors: params.kNeighbors,
              fanout: params.fanout,
              numHops: params.numHops,
              topK: params.topK
            })
          });
          
          if (enhancedResponse.ok) {
            const data = await enhancedResponse.json();
            // Update the results
            setResults(data.relevantTriples || []);
            resultCount = data.count || 0;
            relevanceScore = data.relevanceScore || 0;
            
            // Log the query with performance metrics
            logQuery(query, queryMode, {
              executionTimeMs: Date.now() - startTime,
              relevanceScore,
              resultCount,
              precision: data.precision || 0,
              recall: data.recall || 0,
            });
            
            // Log to console instead of showing alert
            let message = `Enhanced query completed. Found ${resultCount} relevant triples`;
            if (data.metadata?.entityMatches) {
              message += ` from ${data.metadata.entityMatches} matched entities`;
            }
            console.log(message);
            setIsLoading(false);
            return;
          }
        } catch (enhancedError) {
          console.warn('Enhanced query error (falling back to traditional query):', enhancedError);
          // Continue to traditional query as fallback
        }
      }
      
      // Call the traditional backend API as fallback or if explicitly selected
      queryMode = 'traditional';
      const response = await fetch(`/api/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query,
          kNeighbors: params.kNeighbors,
          fanout: params.fanout,
          numHops: params.numHops,
          topK: params.topK,
          queryMode: queryMode, // Explicitly pass the query mode
          useTraditional: true  // Force use of the direct pattern matching approach
        }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to query the RAG backend');
      }
      
      const data = await response.json();
      
      // Update the results
      setResults(data.relevantTriples || []);
      resultCount = data.count || 0;
      relevanceScore = data.relevanceScore || 0;
      
      // Log the query with performance metrics
      logQuery(query, queryMode, {
        executionTimeMs: Date.now() - startTime,
        relevanceScore,
        resultCount,
        precision: data.precision || 0,
        recall: data.recall || 0,
      });
      
      // Log to console instead of showing alert
      let message = `Query completed. Found ${resultCount} relevant triples`;
      if (vectorEnabled && params.useVectorSearch) {
        message += ` (using standard vector search)`;
      }
      console.log(message);
    } catch (error) {
      console.error("RAG query error:", error);
      setErrorMessage(error instanceof Error ? error.message : "An unknown error occurred");
      setResults([]);
      
      // Log failed query
      logQuery(query, queryMode, {
        executionTimeMs: Date.now() - startTime,
        resultCount: 0,
        error: error instanceof Error ? error.message : "Unknown error"
      });
    } finally {
      setIsLoading(false);
    }
  };

  // Helper function to log queries
  const logQuery = async (
    query: string, 
    queryMode: 'pure-rag' | 'vector-search' | 'traditional',
    metrics: {
      executionTimeMs: number;
      relevanceScore?: number;
      precision?: number;
      recall?: number;
      resultCount: number;
      error?: string;
    }
  ) => {
    try {
      await fetch('/api/query-log', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          query,
          queryMode,
          metrics
        })
      });
      console.log('Query logged successfully');
    } catch (error) {
      // Non-blocking error, just log it
      console.warn('Failed to log query:', error);
    }
  };

  const clearResults = () => {
    setResults(null);
    setErrorMessage(null);
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Main Content */}
      <main className="container mx-auto px-6 py-12">
        {/* Header Section */}
        <div className="flex items-center justify-between mb-8">
          <Link href="/" className="inline-flex items-center gap-3 px-4 py-2 text-sm font-medium border border-border/40 hover:border-border/60 bg-background hover:bg-muted/30 rounded-lg transition-colors">
            <ArrowLeft className="h-4 w-4" />
            Back to Documents
          </Link>
        </div>

        {/* Two Column Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Left Column - Database Connections */}
          <div className="lg:col-span-1 space-y-6">
            <div className="nvidia-build-card">
              <DatabaseConnection />
            </div>
            
            {/* Performance Metrics Card */}
            {metrics && (
              <div className="nvidia-build-card">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-3">
                    <div className="w-6 h-6 rounded-md bg-nvidia-green/15 flex items-center justify-center">
                      <BarChart2 className="h-3 w-3 text-nvidia-green" />
                    </div>
                    <h3 className="text-base font-semibold text-foreground">Performance Metrics</h3>
                  </div>
                  <Link href="/rag/metrics" className="text-xs text-nvidia-green hover:text-nvidia-green/80 font-medium underline underline-offset-2">
                    View All
                  </Link>
                </div>
                
                <div className="space-y-3 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Avg. Query Time:</span>
                    <span className="font-medium">{metrics.avgQueryTime > 0 ? `${metrics.avgQueryTime.toFixed(2)}ms` : "No data"}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Relevance Score:</span>
                    <span className="font-medium">{metrics.avgRelevance > 0 ? `${(metrics.avgRelevance * 100).toFixed(1)}%` : "No data"}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Precision:</span>
                    <span className="font-medium">{metrics.precision > 0 ? `${(metrics.precision * 100).toFixed(1)}%` : "No data"}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Recall:</span>
                    <span className="font-medium">{metrics.recall > 0 ? `${(metrics.recall * 100).toFixed(1)}%` : "No data"}</span>
                  </div>
                </div>
              </div>
            )}
          </div>
          
          {/* Right Column - RAG Query Interface */}
          <div className="lg:col-span-3">
            <RagQuery
              onQuerySubmit={handleQuerySubmit}
              clearResults={clearResults}
              isLoading={isLoading}
              error={errorMessage}
              vectorEnabled={vectorEnabled}
            />
            
            {/* Results Section */}
            {results && results.length > 0 && (
              <div className="mt-8 nvidia-build-card">
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-6 h-6 rounded-md bg-nvidia-green/15 flex items-center justify-center">
                    <SearchIcon className="h-3 w-3 text-nvidia-green" />
                  </div>
                  <h3 className="text-lg font-semibold text-foreground">Results ({results.length})</h3>
                </div>
                <div className="space-y-4">
                  {results.map((triple, index) => (
                    <div key={index} className="bg-muted/20 border border-border/20 p-4 rounded-xl">
                      {currentParams.usePureRag ? (
                        // Pure RAG display format (no subject/predicate/object columns)
                        <div className="p-2 rounded">
                          {triple.usedFallback && (
                            <div className="mb-2 text-sm px-3 py-1 bg-amber-500/20 text-amber-700 dark:text-amber-400 rounded-md inline-block">
                              Using general knowledge (no documents found)
                            </div>
                          )}
                          <p className="font-medium break-words">{triple.object}</p>
                        </div>
                      ) : (
                        // Standard triple display for other modes
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                          <div className="bg-background/60 border border-border/30 p-3 rounded-lg">
                            <p className="text-xs font-medium text-nvidia-green uppercase tracking-wider mb-1">Subject</p>
                            <p className="font-medium break-words text-foreground">{triple.subject}</p>
                          </div>
                          <div className="bg-background/60 border border-border/30 p-3 rounded-lg">
                            <p className="text-xs font-medium text-nvidia-green uppercase tracking-wider mb-1">Predicate</p>
                            <p className="font-medium break-words text-foreground">{triple.predicate}</p>
                          </div>
                          <div className="bg-background/60 border border-border/30 p-3 rounded-lg">
                            <p className="text-xs font-medium text-nvidia-green uppercase tracking-wider mb-1">Object</p>
                            <p className="font-medium break-words text-foreground">{triple.object}</p>
                          </div>
                        </div>
                      )}
                      {triple.confidence && !currentParams.usePureRag && (
                        <div className="mt-2 text-xs text-muted-foreground">
                          Confidence: {(triple.confidence * 100).toFixed(1)}%
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
            
            {results && results.length === 0 && !isLoading && (
              <div className="mt-8 nvidia-build-card border-dashed">
                <div className="text-center py-8">
                  <div className="w-12 h-12 rounded-xl bg-muted/30 flex items-center justify-center mx-auto mb-4">
                    <SearchIcon className="h-6 w-6 text-muted-foreground" />
                  </div>
                  <p className="text-foreground font-medium mb-2">No results found for your query</p>
                  <p className="text-sm text-muted-foreground">Try adjusting your query or parameters</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
} 