import { NextResponse } from "next/server";

export async function GET() {
  // Only return the necessary configuration data
  return NextResponse.json({
    nvidiaApiKey: process.env.NVIDIA_API_KEY || null,
    // xaiApiKey removed - integration has been removed
    ollamaBaseUrl: process.env.OLLAMA_BASE_URL || 'http://localhost:11434/v1',
    ollamaModel: process.env.OLLAMA_MODEL || 'qwen3:1.7b',
    vllmBaseUrl: process.env.VLLM_BASE_URL || 'http://localhost:8001/v1',
    vllmModel: process.env.VLLM_MODEL || 'meta-llama/Llama-3.2-3B-Instruct',
    // Add other config values as needed
  });
} 