import { NextRequest, NextResponse } from 'next/server';

/**
 * Legacy Neo4j disconnect endpoint - redirects to the new graph-db/disconnect endpoint
 * @deprecated Use /api/graph-db/disconnect instead with type=neo4j
 */
export async function POST(request: NextRequest) {
  console.log('Redirecting from deprecated /api/neo4j/disconnect to /api/graph-db/disconnect?type=neo4j');
  
  // Create the new URL with the neo4j type parameter
  const url = new URL(request.url);
  const newUrl = new URL('/api/graph-db/disconnect', url.origin);
  
  // Copy all query parameters
  url.searchParams.forEach((value, key) => {
    newUrl.searchParams.append(key, value);
  });
  
  // Add Neo4j type parameter if not present
  if (!newUrl.searchParams.has('type')) {
    newUrl.searchParams.append('type', 'neo4j');
  }
  
  // Clone the request with the new URL
  const newRequest = new Request(newUrl, {
    method: request.method,
    headers: request.headers,
    body: request.body,
    cache: request.cache,
    credentials: request.credentials,
    integrity: request.integrity,
    keepalive: request.keepalive,
    mode: request.mode,
    redirect: request.redirect,
    referrer: request.referrer,
    referrerPolicy: request.referrerPolicy,
    signal: request.signal,
    duplex: 'half',
  } as RequestInit);
  
  // Fetch from the new endpoint
  const response = await fetch(newRequest);
  
  // Return the response
  return response;
} 