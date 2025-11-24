import { NextRequest, NextResponse } from 'next/server'

const PYGRAPHISTRY_SERVICE_URL = process.env.PYGRAPHISTRY_SERVICE_URL || 'http://localhost:8080'

export async function GET(request: NextRequest) {
  try {
    // Forward the request to the PyGraphistry service
    const response = await fetch(`${PYGRAPHISTRY_SERVICE_URL}/api/health`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    })

    if (!response.ok) {
      const errorText = await response.text()
      console.error('PyGraphistry service health check failed:', errorText)
      return NextResponse.json(
        { 
          status: 'error',
          error: 'PyGraphistry service unhealthy', 
          details: errorText 
        },
        { status: response.status }
      )
    }

    const data = await response.json()
    return NextResponse.json(data)
    
  } catch (error) {
    console.error('Error connecting to PyGraphistry service:', error)
    return NextResponse.json(
      { 
        status: 'error',
        error: 'Failed to connect to PyGraphistry service',
        details: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    )
  }
} 