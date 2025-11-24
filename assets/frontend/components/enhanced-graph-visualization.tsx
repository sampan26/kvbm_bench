"use client"

import React, { useState, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Switch } from '@/components/ui/switch'
import { Zap, Cpu, Eye, Settings } from 'lucide-react'
import { ForceGraphWrapper } from './force-graph-wrapper'
import { PyGraphistryViewer } from './pygraphistry-viewer'
import type { Triple } from '@/utils/text-processing'

interface GraphData {
  nodes: Array<{
    id: string
    name: string
    group?: string
    [key: string]: any
  }>
  links: Array<{
    source: string
    target: string
    name: string
    [key: string]: any
  }>
}

interface EnhancedGraphVisualizationProps {
  graphData?: GraphData
  jsonData?: any // For backward compatibility with existing ForceGraphWrapper
  triples?: Triple[]
  fullscreen?: boolean
  layoutType?: string
  highlightedNodes?: string[]
  onError?: (error: Error) => void
}

export function EnhancedGraphVisualization({
  graphData,
  jsonData,
  triples,
  fullscreen = false,
  layoutType,
  highlightedNodes,
  onError
}: EnhancedGraphVisualizationProps) {
  const [activeTab, setActiveTab] = useState<'threejs' | 'pygraphistry'>('threejs')
  const [gpuPreferred, setGpuPreferred] = useState(false)
  
  // Convert triples to graph data format if needed
  const processedGraphData = React.useMemo(() => {
    if (graphData) {
      return graphData
    }
    
    if (triples && triples.length > 0) {
      const nodes = new Map<string, any>()
      const links: any[] = []
      
      triples.forEach((triple, index) => {
        // Triple interface has simple string properties
        const subjectId = triple.subject
        const subjectName = triple.subject
        const objectId = triple.object
        const objectName = triple.object
        const predicateName = triple.predicate
        
        // Add nodes
        if (!nodes.has(subjectId)) {
          nodes.set(subjectId, {
            id: subjectId,
            name: subjectName,
            group: 'entity'
          })
        }
        
        if (!nodes.has(objectId)) {
          nodes.set(objectId, {
            id: objectId,
            name: objectName,
            group: 'entity'
          })
        }
        
        // Add link
        links.push({
          source: subjectId,
          target: objectId,
          name: predicateName
        })
      })
      
      return {
        nodes: Array.from(nodes.values()),
        links: links
      }
    }
    
    return null
  }, [graphData, triples])

  const handleTabChange = useCallback((value: string) => {
    setActiveTab(value as 'threejs' | 'pygraphistry')
  }, [])

  const handleError = useCallback((error: Error) => {
    console.error('Graph visualization error:', error)
    if (onError) {
      onError(error)
    }
  }, [onError])

  const nodeCount = processedGraphData?.nodes?.length || jsonData?.nodes?.length || 0
  const linkCount = processedGraphData?.links?.length || jsonData?.links?.length || 0

  return (
    <div className="w-full h-full">
      <Card className="h-full">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <Eye className="w-5 h-5" />
              Knowledge Graph Visualization
            </CardTitle>
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <span>{nodeCount} nodes</span>
                <span>•</span>
                <span>{linkCount} edges</span>
              </div>
              <div className="flex items-center space-x-2">
                <Switch
                  id="gpu-preferred"
                  checked={gpuPreferred}
                  onCheckedChange={(checked) => {
                    setGpuPreferred(checked)
                    if (checked && activeTab === 'threejs') {
                      setActiveTab('pygraphistry')
                    }
                  }}
                />
                <label htmlFor="gpu-preferred" className="text-sm font-medium">
                  GPU Preferred
                </label>
              </div>
            </div>
          </div>
        </CardHeader>
        <CardContent className="p-0 h-[calc(100%-80px)]">
          <Tabs value={activeTab} onValueChange={handleTabChange} className="h-full">
            <div className="px-6 pb-4">
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="threejs" className="flex items-center gap-2">
                  <Cpu className="w-4 h-4" />
                  Client-Side (Three.js)
                  <Badge variant="secondary" className="ml-1">WebGPU</Badge>
                </TabsTrigger>
                <TabsTrigger value="pygraphistry" className="flex items-center gap-2">
                  <Zap className="w-4 h-4" />
                  Server-Side (PyGraphistry)
                  <Badge variant="secondary" className="ml-1">GPU</Badge>
                </TabsTrigger>
              </TabsList>
            </div>
            
            <TabsContent value="threejs" className="h-[calc(100%-80px)] px-6 pb-6 mt-0">
              <div className="h-full rounded-lg border overflow-hidden">
                <ForceGraphWrapper
                  jsonData={jsonData || {
                    nodes: processedGraphData?.nodes || [],
                    links: processedGraphData?.links || []
                  }}
                  fullscreen={fullscreen}
                  layoutType={layoutType}
                  highlightedNodes={highlightedNodes}
                  onError={handleError}
                />
              </div>
            </TabsContent>
            
            <TabsContent value="pygraphistry" className="h-[calc(100%-80px)] px-6 pb-6 mt-0">
              <div className="h-full">
                {processedGraphData ? (
                  <PyGraphistryViewer
                    graphData={processedGraphData}
                    onError={handleError}
                  />
                ) : (
                  <div className="h-full flex items-center justify-center border rounded-lg bg-muted/50">
                    <div className="text-center space-y-2">
                      <div className="text-muted-foreground">
                        No graph data available for PyGraphistry visualization
                      </div>
                      <div className="text-sm text-muted-foreground">
                        Please load graph data to enable GPU-accelerated visualization
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  )
} 