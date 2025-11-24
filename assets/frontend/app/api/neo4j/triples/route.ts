import { NextRequest, NextResponse } from 'next/server';

/**
 * Legacy Neo4j triples endpoint - redirects to the new graph-db/triples endpoint
 * @deprecated Use /api/graph-db/triples instead with type=neo4j
 */
export async function GET(req: NextRequest) {
  console.log('Redirecting from deprecated /api/neo4j/triples to /api/graph-db/triples?type=neo4j');
  
  // Create the new URL with the same query parameters
  const url = new URL(req.url);
  const newUrl = new URL('/api/graph-db/triples', url.origin);
  
  // Copy all query parameters
  url.searchParams.forEach((value, key) => {
    newUrl.searchParams.append(key, value);
  });
  
  // Add Neo4j type parameter if not present
  if (!newUrl.searchParams.has('type')) {
    newUrl.searchParams.append('type', 'neo4j');
  }
  
  // Return a redirect response
  return NextResponse.redirect(newUrl);
}

/**
 * Legacy Neo4j triples POST endpoint - redirects to the new graph-db/triples endpoint
 * @deprecated Use /api/graph-db/triples instead with type=neo4j
 */
export async function POST(req: NextRequest) {
  console.log('Redirecting from deprecated /api/neo4j/triples to /api/graph-db/triples?type=neo4j');
  
  // Create the new URL with the neo4j type parameter
  const url = new URL(req.url);
  const newUrl = new URL('/api/graph-db/triples', url.origin);
  
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
    method: req.method,
    headers: req.headers,
    body: req.body,
    cache: req.cache,
    credentials: req.credentials,
    integrity: req.integrity,
    keepalive: req.keepalive,
    mode: req.mode,
    redirect: req.redirect,
    referrer: req.referrer,
    referrerPolicy: req.referrerPolicy,
    signal: req.signal,
    duplex: 'half',
  } as RequestInit);
  
  // Fetch from the new endpoint
  const response = await fetch(newRequest);
  
  // Return the response
  return response;
} 