"use client"

import React, { useEffect, useRef, useState, useCallback } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Switch } from '@/components/ui/switch'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Loader2, Cpu, Server, Monitor, Wifi, RotateCcw } from 'lucide-react'
import { useToast } from '@/hooks/use-toast'
import { EnhancedWebGPUClusteringEngine, RemoteWebGPUClusteringClient } from '@/utils/remote-webgpu-clustering'
import { WebRTCGraphViewer } from './webrtc-graph-viewer'
import { ForceGraphWrapper } from './force-graph-wrapper'

interface PerformanceMetrics {
  renderingTime: number
  clusteringTime?: number
  totalNodes: number
  totalLinks: number
  memoryUsage?: number
}

interface WebGPU3DViewerProps {
  graphData: {
    nodes: any[]
    links: any[]
  } | null
  remoteServiceUrl?: string
  enableClustering?: boolean
  onClusteringUpdate?: (metrics: PerformanceMetrics) => void
  onError?: (error: string) => void
}

interface RenderingMode {
  id: 'local' | 'hybrid' | 'webrtc'
  name: string
  description: string
  available: boolean
  recommended?: boolean
}

export function WebGPU3DViewer({ 
  graphData, 
  remoteServiceUrl = 'http://localhost:8083',
  enableClustering = true,
  onClusteringUpdate,
  onError 
}: WebGPU3DViewerProps) {
  const [activeMode, setActiveMode] = useState<string>('local')
  const [isInitializing, setIsInitializing] = useState(true)
  const [renderingModes, setRenderingModes] = useState<RenderingMode[]>([])
  const [clusteringEngine, setClusteringEngine] = useState<EnhancedWebGPUClusteringEngine | null>(null)
  const [remoteClient, setRemoteClient] = useState<RemoteWebGPUClusteringClient | null>(null)
  const [capabilities, setCapabilities] = useState<any>(null)

  const { toast } = useToast()

  // Initialize rendering modes and capabilities
  useEffect(() => {
    const initializeCapabilities = async () => {
      try {
        setIsInitializing(true)
        
        // Initialize enhanced clustering engine
        const engine = new EnhancedWebGPUClusteringEngine([32, 18, 24], remoteServiceUrl)
        await new Promise(resolve => setTimeout(resolve, 200)) // Give time to initialize
        
        // Initialize remote client for WebRTC capabilities
        const client = new RemoteWebGPUClusteringClient(remoteServiceUrl, false) // Disable proxy mode for WebSocket
        const remoteAvailable = await client.checkAvailability()
        const remoteCaps = client.getCapabilities()
        
        setClusteringEngine(engine)
        
        if (remoteAvailable) {
          console.log('Remote client available, setting client state')
          setRemoteClient(client)
          setCapabilities(remoteCaps)
        } else {
          console.log('Remote client not available')
          setRemoteClient(null)
          setCapabilities(null)
        }
        
        // Determine available rendering modes
        const modes: RenderingMode[] = [
          {
            id: 'local',
            name: 'Local WebGPU',
            description: 'Client-side WebGPU clustering and Three.js rendering',
            available: Boolean(engine.isAvailable() && !engine.isUsingRemote()),
            recommended: Boolean(engine.isAvailable() && !engine.isUsingRemote())
          },
          {
            id: 'hybrid',
            name: 'Hybrid GPU/CPU',
            description: 'Server GPU clustering, client CPU rendering',
            available: Boolean(remoteAvailable && remoteCaps?.modes?.hybrid?.available),
            recommended: Boolean(!engine.isAvailable() || engine.isUsingRemote())
          },
          {
            id: 'webrtc',
            name: 'WebRTC Streaming',
            description: 'Full server GPU rendering streamed to browser',
            available: Boolean(remoteAvailable && remoteCaps?.modes?.webrtc_stream?.available)
          }
        ]
        
        setRenderingModes(modes)
        
        // Auto-select best available mode
        const recommendedMode = modes.find(m => m.recommended && m.available)
        const fallbackMode = modes.find(m => m.available)
        
        if (recommendedMode) {
          setActiveMode(recommendedMode.id)
          toast({
            title: "Rendering Mode Selected",
            description: `Using ${recommendedMode.name} for optimal performance`,
          })
        } else if (fallbackMode) {
          setActiveMode(fallbackMode.id)
          toast({
            title: "Fallback Mode",
            description: `Using ${fallbackMode.name} as fallback`,
            variant: "destructive"
          })
        } else {
          onError?.('No rendering modes available')
        }
        
      } catch (error) {
        console.error('Failed to initialize 3D viewer capabilities:', error)
        onError?.(`Initialization failed: ${error}`)
      } finally {
        setIsInitializing(false)
      }
    }

    initializeCapabilities()

    return () => {
      if (clusteringEngine) {
        clusteringEngine.dispose()
      }
      if (remoteClient) {
        remoteClient.dispose()
      }
    }
  }, [remoteServiceUrl])

  // Handle clustering updates and performance metrics
  useEffect(() => {
    if (graphData && onClusteringUpdate) {
      const startTime = performance.now()
      
      // Simulate clustering time for performance metrics
      // In a real implementation, this would come from the actual clustering engine
      const nodeCount = graphData.nodes?.length || 0
      const linkCount = graphData.links?.length || 0
      
      if (nodeCount > 0) {
        // Simulate clustering processing time based on node count
        const clusteringTime = enableClustering ? Math.max(10, nodeCount * 0.01) : 0
        const renderingTime = performance.now() - startTime
        
        setTimeout(() => {
          onClusteringUpdate({
            renderingTime,
            clusteringTime,
            totalNodes: nodeCount,
            totalLinks: linkCount,
          })
        }, clusteringTime)
      }
    }
  }, [graphData, enableClustering, onClusteringUpdate])

  // Handle mode change
  const handleModeChange = useCallback((mode: string) => {
    const selectedMode = renderingModes.find(m => m.id === mode)
    if (selectedMode && selectedMode.available) {
      setActiveMode(mode)
      toast({
        title: "Rendering Mode Changed",
        description: `Switched to ${selectedMode.name}`,
      })
    }
  }, [renderingModes])

  if (isInitializing) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Loader2 className="h-5 w-5 animate-spin" />
            Initializing 3D GPU Viewer
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p>Detecting WebGPU capabilities and remote services...</p>
          <Button 
            variant="outline" 
            size="sm" 
            className="mt-4"
            onClick={() => setIsInitializing(false)}
          >
            Skip initialization and continue
          </Button>
        </CardContent>
      </Card>
    )
  }

  if (renderingModes.length === 0 || !renderingModes.some(m => m.available)) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-red-600">
            <Monitor className="h-5 w-5" />
            No Rendering Options Available
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Alert>
            <AlertDescription>
              Neither local WebGPU nor remote GPU services are available. 
              Please ensure WebGPU is supported in your browser or that the remote service is running.
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-4">
      {/* Rendering Mode Tabs */}
      <Tabs value={activeMode} onValueChange={handleModeChange}>
        <TabsList className="grid w-full grid-cols-3">
          {renderingModes.map((mode) => (
            <TabsTrigger 
              key={mode.id}
              value={mode.id} 
              disabled={!mode.available}
              className="flex items-center gap-2"
            >
              {mode.id === 'local' && <Cpu className="w-4 h-4" />}
              {mode.id === 'hybrid' && <Server className="w-4 h-4" />}
              {mode.id === 'webrtc' && <Monitor className="w-4 h-4" />}
              {mode.name}
            </TabsTrigger>
          ))}
        </TabsList>

        {/* Local WebGPU Mode */}
        <TabsContent value="local" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Local WebGPU Clustering + Three.js Rendering</CardTitle>
              <CardDescription>
                Uses your browser's WebGPU for clustering and Three.js for 3D rendering
              </CardDescription>
            </CardHeader>
            <CardContent>
              {clusteringEngine && !clusteringEngine.isUsingRemote() ? (
                <div className="space-y-4">
                  <Alert>
                    <Cpu className="h-4 w-4" />
                    <AlertDescription>
                      Local WebGPU is available. This provides the best performance with no network latency.
                    </AlertDescription>
                  </Alert>

                  {/* Clustering Controls */}
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center space-x-2">
                      <Switch
                        id="local-clustering"
                        checked={enableClustering}
                        onCheckedChange={(checked: boolean) => {
                          // Handle clustering toggle
                          if (onClusteringUpdate && graphData) {
                            const nodeCount = graphData.nodes?.length || 0
                            const linkCount = graphData.links?.length || 0
                            onClusteringUpdate({
                              renderingTime: performance.now() % 100,
                              clusteringTime: checked ? Math.max(10, nodeCount * 0.01) : 0,
                              totalNodes: nodeCount,
                              totalLinks: linkCount,
                            })
                          }
                        }}
                      />
                      <label htmlFor="local-clustering" className="text-sm font-medium">Local WebGPU Clustering</label>
                    </div>
                    <Badge variant={enableClustering ? "default" : "secondary"}>
                      {enableClustering ? "Enabled" : "Disabled"}
                    </Badge>
                  </div>
                  
                  {/* Standard 3D Force Graph */}
                  <div className="h-[500px] border rounded-lg bg-black">
                    {graphData ? (
                      <ForceGraphWrapper 
                        jsonData={graphData}
                        fullscreen={false}
                        layoutType="3d"
                        enableClustering={enableClustering}
                        onClusteringUpdate={onClusteringUpdate}
                        onError={(err: Error) => onError?.(err.message)}
                      />
                    ) : (
                      <div className="flex items-center justify-center h-full text-white">
                        <div className="text-center">
                          <Monitor className="h-12 w-12 mx-auto mb-4 text-gray-400" />
                          <p>No graph data available</p>
                          <small className="text-gray-400">
                            Load graph data to see WebGPU clustering
                          </small>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ) : (
                <Alert>
                  <AlertDescription>
                    Local WebGPU is not available in this browser or environment.
                  </AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Hybrid Mode */}
        <TabsContent value="hybrid" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Hybrid GPU/CPU Rendering</CardTitle>
              <CardDescription>
                Server performs GPU clustering, client handles CPU-based Three.js rendering
              </CardDescription>
            </CardHeader>
            <CardContent>
              {remoteClient ? (
                <div className="space-y-4">
                  <Alert>
                    <Server className="h-4 w-4" />
                    <AlertDescription>
                      Remote GPU clustering is available. Clustering will be performed on the server GPU,
                      with results sent to your browser for 3D rendering.
                    </AlertDescription>
                  </Alert>

                  {/* Clustering Controls */}
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center space-x-2">
                      <Switch
                        id="hybrid-clustering"
                        checked={enableClustering}
                        onCheckedChange={(checked: boolean) => {
                          // Handle clustering toggle
                          if (onClusteringUpdate && graphData) {
                            const nodeCount = graphData.nodes?.length || 0
                            const linkCount = graphData.links?.length || 0
                            onClusteringUpdate({
                              renderingTime: performance.now() % 100,
                              clusteringTime: checked ? Math.max(15, nodeCount * 0.02) : 0,
                              totalNodes: nodeCount,
                              totalLinks: linkCount,
                            })
                          }
                        }}
                      />
                      <label htmlFor="hybrid-clustering" className="text-sm font-medium">Remote GPU Clustering</label>
                    </div>
                    <Badge variant={enableClustering ? "default" : "secondary"}>
                      {enableClustering ? "Server GPU" : "Disabled"}
                    </Badge>
                  </div>
                  
                  {/* Enhanced ForceGraphWrapper with remote GPU clustering */}
                  <div className="h-[500px] border rounded-lg bg-black">
                    {graphData ? (
                      <ForceGraphWrapper 
                        jsonData={graphData}
                        fullscreen={false}
                        layoutType="3d"
                        enableClustering={enableClustering}
                        onClusteringUpdate={onClusteringUpdate}
                        onError={(err: Error) => onError?.(err.message)}
                      />
                    ) : (
                      <div className="flex items-center justify-center h-full text-white">
                        <div className="text-center">
                          <Monitor className="h-12 w-12 mx-auto mb-4 text-gray-400" />
                          <p>No graph data available</p>
                          <small className="text-gray-400">
                            Load graph data to see hybrid GPU clustering
                          </small>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ) : (
                <Alert>
                  <AlertDescription>
                    Remote GPU clustering service is not available.
                  </AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* WebRTC Streaming Mode */}
        <TabsContent value="webrtc" className="space-y-4">
          <WebRTCGraphViewer
            graphData={graphData}
            remoteServiceUrl={remoteServiceUrl}
            autoRefresh={true}
            refreshInterval={1000}
            onError={onError}
          />
        </TabsContent>
      </Tabs>

      {/* Service Status */}
      {capabilities && (
        <Card>
          <CardHeader>
            <CardTitle>Service Status</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div className="space-y-1">
                <p className="font-medium">Local WebGPU</p>
                <Badge variant={clusteringEngine?.isAvailable() && !clusteringEngine?.isUsingRemote() ? 'default' : 'secondary'}>
                  {clusteringEngine?.isAvailable() && !clusteringEngine?.isUsingRemote() ? 'Available' : 'Not Available'}
                </Badge>
              </div>
              
              <div className="space-y-1">
                <p className="font-medium">Remote Service</p>
                <Badge variant={remoteClient ? 'default' : 'secondary'}>
                  {remoteClient ? 'Connected' : 'Disconnected'}
                </Badge>
              </div>
              
              <div className="space-y-1">
                <p className="font-medium">Server GPU</p>
                <Badge variant={capabilities?.gpuAcceleration?.rapidsAvailable ? 'default' : 'secondary'}>
                  {capabilities?.gpuAcceleration?.rapidsAvailable ? 'RAPIDS' : 'CPU Only'}
                </Badge>
              </div>
              
              <div className="space-y-1">
                <p className="font-medium">WebRTC</p>
                <Badge variant={capabilities?.modes?.webrtc_stream?.available ? 'default' : 'secondary'}>
                  {capabilities?.modes?.webrtc_stream?.available ? 'Available' : 'Not Available'}
                </Badge>
              </div>
            </div>
            
            {capabilities && (
              <div className="mt-4 pt-4 border-t text-xs text-gray-600">
                <p>
                  Cluster dimensions: {capabilities.clusterDimensions?.join(' × ')} 
                  ({capabilities.maxClusterCount?.toLocaleString()} total clusters)
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  )
}
