import { NextRequest, NextResponse } from 'next/server';
import neo4jService from '@/lib/neo4j';

/**
 * API endpoint to create a test query log
 * This is for debugging purposes only
 */
export async function GET(request: NextRequest) {
  try {
    console.log('[Test] Creating test query log');
    
    // Initialize Neo4j if not already
    if (!neo4jService.isInitialized()) {
      console.log('[Test] Initializing Neo4j service');
      neo4jService.initialize();
    }
    
    // Get query text from URL parameters or use a default
    const query = request.nextUrl.searchParams.get('query') || 'Test query for debugging';
    const queryMode = (request.nextUrl.searchParams.get('mode') || 'traditional') as 'traditional' | 'vector-search' | 'pure-rag';
    const executionTime = parseInt(request.nextUrl.searchParams.get('time') || '300');
    const resultCount = parseInt(request.nextUrl.searchParams.get('count') || '5');
    
    console.log(`[Test] Adding test query: "${query}" (${queryMode})`);
    
    // Log the query with some test metrics
    await neo4jService.logQuery(
      query,
      queryMode,
      {
        executionTimeMs: executionTime,
        relevanceScore: 0,
        precision: 0,
        recall: 0,
        resultCount: resultCount
      }
    );
    
    // Get current query logs to verify
    const logs = await neo4jService.getQueryLogs(10);
    
    return NextResponse.json({
      success: true,
      message: `Test query "${query}" added successfully`,
      logs: logs.slice(0, 3) // Return top 3 logs for verification
    });
  } catch (error) {
    console.error('[Test] Error creating test query log:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json(
      { error: errorMessage },
      { status: 500 }
    );
  }
} 