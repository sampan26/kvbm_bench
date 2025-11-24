import { NextRequest, NextResponse } from 'next/server';
import backendService from '@/lib/backend-service';
import type { Triple } from '@/types/graph';
import { getGraphDbType } from '../settings/route';

export async function POST(request: NextRequest) {
  try {
    const { query, triples, kNeighbors, fanout, numHops, useTraditional, queryMode } = await request.json();
    
    if (!query) {
      return NextResponse.json({ error: 'Query is required' }, { status: 400 });
    }
    
    // Initialize backend if needed with the selected graph DB type
    if (!backendService.isInitialized) {
      const graphDbType = getGraphDbType();
      console.log(`Initializing backend with graph DB type: ${graphDbType}`);
      await backendService.initialize(graphDbType);
    }
    
    // Process triples if provided
    if (triples && Array.isArray(triples) && triples.length > 0) {
      await backendService.processTriples(triples);
    }
    
    // Determine if we should use traditional search based on queryMode
    // This allows the frontend to explicitly choose traditional search
    const shouldUseTraditional = useTraditional || (queryMode === 'traditional');
    
    console.log(`Query mode: ${queryMode}, Using traditional search: ${shouldUseTraditional}`);
    
    // Query the backend
    const relevantTriples = await backendService.query(
      query,
      kNeighbors || 4096,
      fanout || 400,
      numHops || 2,
      shouldUseTraditional // Pass the flag to use traditional search
    );
    
    // Return results
    return NextResponse.json({
      query,
      relevantTriples,
      count: relevantTriples.length,
      message: `Found ${relevantTriples.length} relevant triples for query: "${query}"${shouldUseTraditional ? ' using traditional search' : ''}`
    });
  } catch (error) {
    console.error('Error querying backend:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json({ error: errorMessage }, { status: 500 });
  }
} 