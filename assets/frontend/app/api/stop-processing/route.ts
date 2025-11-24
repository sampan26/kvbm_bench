import { NextRequest, NextResponse } from 'next/server'

// Global flag to track if processing should be stopped
let shouldStopProcessing = false

// Function to check if processing should stop
export function getShouldStopProcessing(): boolean {
  return shouldStopProcessing
}

// Function to reset the stop flag
export function resetStopProcessing(): void {
  shouldStopProcessing = false
}

// Function to set the stop flag
export function setStopProcessing(): void {
  shouldStopProcessing = true
}

export async function POST(request: NextRequest) {
  try {
    console.log('Stop processing request received')
    
    // Set the global flag to stop processing
    shouldStopProcessing = true
    
    // You could also implement more sophisticated cancellation here
    // such as canceling ongoing HTTP requests, clearing queues, etc.
    
    return NextResponse.json({ 
      success: true, 
      message: 'Processing stop signal sent' 
    })
  } catch (error) {
    console.error('Error stopping processing:', error)
    return NextResponse.json(
      { error: 'Failed to stop processing' },
      { status: 500 }
    )
  }
}

export async function GET(request: NextRequest) {
  // Allow checking the current stop status
  return NextResponse.json({ 
    shouldStop: shouldStopProcessing 
  })
}
