# GPU Graph Visualization Services

## 🚀 Overview

This directory contains optional GPU-accelerated graph visualization services that run separately from the main txt2kg application. These services provide advanced visualization capabilities for large-scale graphs.

**Note**: These services are **optional** and not included in the default docker-compose configurations. They must be run separately.

## 📦 Available Services

### 1. Unified GPU Service (`unified_gpu_service.py`)
Provides **Local GPU (cuGraph)** and **CPU (NetworkX)** processing in a single FastAPI service.

**Processing Modes:**
| Mode | Description | Requirements |
|------|-------------|--------------|
| **Local GPU (cuGraph)** | Full GPU processing on your hardware | NVIDIA GPU + cuGraph |
| **Local CPU** | NetworkX fallback processing | None |

### 2. Remote GPU Rendering Service (`remote_gpu_rendering_service.py`)
Provides GPU-accelerated graph layout and rendering with iframe-embeddable visualizations.

### 3. Local GPU Service (`local_gpu_viz_service.py`)
Local GPU processing service with WebSocket support for real-time updates.

## 🛠️ Setup

### Prerequisites
- NVIDIA GPU with CUDA support (for GPU mode)
- RAPIDS cuGraph (for local GPU processing)

### Installation

```bash
# Install dependencies
pip install -r deploy/services/gpu-viz/requirements.txt

# For remote WebGPU service
pip install -r deploy/services/gpu-viz/requirements-remote-webgpu.txt
```

### Running Services

#### Unified GPU Service
```bash
cd deploy/services/gpu-viz
python unified_gpu_service.py
```

Service runs on: http://localhost:8080

#### Remote GPU Rendering Service
```bash
cd deploy/services/gpu-viz
python remote_gpu_rendering_service.py
```

Service runs on: http://localhost:8082

#### Using Startup Script
```bash
cd deploy/services/gpu-viz
./start_remote_gpu_services.sh
```

## 📡 API Usage

### Process Graph with Mode Selection

```bash
curl -X POST http://localhost:8080/api/visualize \
  -H "Content-Type: application/json" \
  -d '{
    "graph_data": {
      "nodes": [{"id": "1", "name": "Node 1"}, {"id": "2", "name": "Node 2"}],
      "links": [{"source": "1", "target": "2", "name": "edge_1_2"}]
    },
    "processing_mode": "local_gpu",
    "layout_algorithm": "force_atlas2",
    "clustering_algorithm": "leiden",
    "compute_centrality": true
  }'
```

### Check Available Capabilities

```bash
curl http://localhost:8080/api/capabilities
```

Response:
```json
{
  "processing_modes": {
    "local_gpu": {"available": true, "description": "Local GPU processing with cuGraph/RAPIDS"},
    "local_cpu": {"available": true, "description": "Local CPU fallback processing with NetworkX"}
  },
  "has_rapids": true,
  "gpu_available": true
}
```

## 🎯 Frontend Integration

The txt2kg frontend includes built-in components for GPU visualization:

- `UnifiedGPUViewer`: Connects to unified GPU service
- `ForceGraphWrapper`: Three.js WebGPU visualization (default)

### Using GPU Services in Frontend

The frontend has API routes that can connect to these services:
- `/api/unified-gpu/*`: Unified GPU service integration

To use these services, ensure they are running separately and configure the frontend environment variables accordingly.

### Processing Graph Data

```javascript
// Local GPU mode  
const response = await fetch('/api/unified-gpu/visualize', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    graph_data: { nodes, links },
    processing_mode: 'local_gpu',
    layout_algorithm: 'force_atlas2',
    clustering_algorithm: 'leiden',
    compute_centrality: true
  })
})

// Local CPU mode  
const response = await fetch('/api/unified-gpu/visualize', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    graph_data: { nodes, links },
    processing_mode: 'local_cpu'
  })
})
```

## 🔧 Configuration Options

### Local GPU Mode  
- `layout_algorithm`: "force_atlas2", "spectral", "fruchterman_reingold"
- `clustering_algorithm`: "leiden", "louvain", "spectral"
- `compute_centrality`: true/false

### Local CPU Mode
- Basic processing with NetworkX fallback
- No additional configuration needed

## 📊 Response Format

```json
{
  "processed_nodes": [...],
  "processed_edges": [...],
  "processing_mode": "local_gpu",
  "layout_positions": {...},
  "clusters": {...},
  "centrality": {...},
  "stats": {
    "node_count": 1000,
    "edge_count": 5000,
    "gpu_accelerated": true,
    "layout_computed": true,
    "clusters_computed": true
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## 🚀 Benefits

### ✅ Advantages
- **Single service** - One port, one deployment
- **Mode switching** - Choose best processing per graph
- **Fallback handling** - Graceful degradation if GPU unavailable  
- **Consistent API** - Same interface for all modes
- **No GPL dependencies** - All dependencies are permissively licensed

### 🎯 Use Cases
- **Local GPU**: Private data, large-scale processing, GPU-accelerated algorithms
- **Local CPU**: Development, testing, small graphs

## 🐛 Troubleshooting

### GPU Not Detected
```bash
# Check GPU availability
nvidia-smi

# Check RAPIDS installation
python -c "import cudf, cugraph; print('RAPIDS OK')"
```

### Service Health
```bash
curl http://localhost:8080/api/health
```

## 📈 Performance Tips

1. **Large graphs (>100k nodes)**: Use `local_gpu` mode with RAPIDS cuGraph
2. **Development**: Use `local_cpu` mode for speed and simplicity
3. **Mixed workloads**: Switch modes dynamically based on graph size and GPU availability

## 📝 License Compliance

This service has been updated to remove all GPL-licensed dependencies:
- ❌ Removed: `igraph` (GPL v2+)
- ❌ Removed: `graphistry` with `compute_igraph` (uses igraph internally)
- ✅ Uses only: NetworkX (BSD), cuGraph (Apache 2.0), NumPy (BSD), pandas (BSD) 