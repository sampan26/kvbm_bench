import { NextRequest, NextResponse } from 'next/server'

const PYGRAPHISTRY_SERVICE_URL = process.env.PYGRAPHISTRY_SERVICE_URL || 'http://localhost:8080'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    
    // Forward the request to the PyGraphistry service
    const response = await fetch(`${PYGRAPHISTRY_SERVICE_URL}/api/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    })

    if (!response.ok) {
      const errorText = await response.text()
      console.error('PyGraphistry service error:', errorText)
      return NextResponse.json(
        { 
          error: 'PyGraphistry service error', 
          details: errorText 
        },
        { status: response.status }
      )
    }

    const data = await response.json()
    return NextResponse.json(data)
    
  } catch (error) {
    console.error('Error forwarding to PyGraphistry service:', error)
    return NextResponse.json(
      { 
        error: 'Failed to connect to PyGraphistry service',
        details: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    )
  }
} 