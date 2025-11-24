import { NextRequest, NextResponse } from 'next/server'

// Simple test endpoint to verify proxy connectivity
const REMOTE_WEBGPU_SERVICE_URL = process.env.REMOTE_WEBGPU_SERVICE_URL || 'http://txt2kg-remote-webgpu:8083'

export async function GET() {
  try {
    console.log(`Testing connection to: ${REMOTE_WEBGPU_SERVICE_URL}`)
    
    const response = await fetch(`${REMOTE_WEBGPU_SERVICE_URL}/health`)
    
    if (!response.ok) {
      throw new Error(`Service responded with ${response.status}: ${response.statusText}`)
    }
    
    const data = await response.json()
    
    return NextResponse.json({
      success: true,
      service_url: REMOTE_WEBGPU_SERVICE_URL,
      service_response: data
    })
    
  } catch (error) {
    console.error('Proxy test failed:', error)
    return NextResponse.json({
      success: false,
      error: String(error),
      service_url: REMOTE_WEBGPU_SERVICE_URL
    }, { status: 500 })
  }
}
