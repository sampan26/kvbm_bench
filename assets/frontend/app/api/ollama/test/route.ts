import { NextRequest, NextResponse } from 'next/server';

/**
 * Test endpoint for Ollama integration
 * GET /api/ollama/test - Test Ollama functionality with sample data
 */

export async function GET(req: NextRequest) {
  try {
    const sampleText = `
    Apple Inc. is a multinational technology company headquartered in Cupertino, California. 
    The company was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976. 
    Apple designs and develops consumer electronics, computer software, and online services. 
    Tim Cook is the current CEO of Apple Inc.
    `;

    console.log('Testing Ollama with sample text...');

    // Test connection first
    const connectionResponse = await fetch(`${req.nextUrl.origin}/api/ollama?action=test-connection`);
    const connectionResult = await connectionResponse.json();

    if (!connectionResult.connected) {
      return NextResponse.json({
        success: false,
        error: 'Ollama connection failed',
        details: connectionResult.error,
        connectionTest: connectionResult
      });
    }

    // Test triple extraction
    const extractionResponse = await fetch(`${req.nextUrl.origin}/api/ollama`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text: sampleText.trim(),
        model: 'qwen3:1.7b',
        temperature: 0.1,
        maxTokens: 1024
      })
    });

    if (!extractionResponse.ok) {
      const errorText = await extractionResponse.text();
      return NextResponse.json({
        success: false,
        error: 'Triple extraction failed',
        details: errorText,
        connectionTest: connectionResult
      });
    }

    const extractionResult = await extractionResponse.json();

    return NextResponse.json({
      success: true,
      message: 'Ollama integration test completed successfully',
      connectionTest: connectionResult,
      extractionTest: {
        inputText: sampleText.trim(),
        triplesExtracted: extractionResult.triples?.length || 0,
        sampleTriples: (extractionResult.triples || []).slice(0, 3),
        method: extractionResult.method,
        model: extractionResult.model
      },
      fullResult: extractionResult
    });
  } catch (error) {
    console.error('Error in Ollama test:', error);
    return NextResponse.json(
      { 
        success: false,
        error: 'Test failed with exception',
        details: error instanceof Error ? error.message : String(error)
      },
      { status: 500 }
    );
  }
}
