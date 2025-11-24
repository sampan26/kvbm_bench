import { NextRequest, NextResponse } from 'next/server'

const UNIFIED_GPU_SERVICE_URL = process.env.UNIFIED_GPU_SERVICE_URL || 'http://localhost:8080'

export async function GET(request: NextRequest) {
  try {
    // Forward the request to the unified GPU service
    const response = await fetch(`${UNIFIED_GPU_SERVICE_URL}/api/capabilities`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    })

    if (!response.ok) {
      const errorText = await response.text()
      console.error('Unified GPU service error:', errorText)
      return NextResponse.json(
        { 
          error: 'Unified GPU service error', 
          details: errorText 
        },
        { status: response.status }
      )
    }

    const data = await response.json()
    return NextResponse.json(data)
    
  } catch (error) {
    console.error('Error forwarding to unified GPU service:', error)
    return NextResponse.json(
      { 
        error: 'Failed to connect to unified GPU service',
        details: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    )
  }
} 