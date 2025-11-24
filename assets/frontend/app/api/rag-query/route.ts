import { NextRequest, NextResponse } from 'next/server';
import RAGService from '@/lib/rag';

/**
 * API endpoint for RAG-based question answering
 * Uses Pinecone for document retrieval and LangChain for generation
 * POST /api/rag-query
 */
export async function POST(req: NextRequest) {
  try {
    // Parse request body
    const body = await req.json();
    const { query, topK = 5 } = body;

    if (!query || typeof query !== 'string') {
      return NextResponse.json({ error: 'Query is required' }, { status: 400 });
    }

    // Initialize the RAG service
    const ragService = RAGService;
    await ragService.initialize();
    
    console.log(`Processing RAG query: "${query}" with topK=${topK}`);

    // Retrieve documents and generate answer
    const answer = await ragService.retrievalQA(query, topK);
    
    // Check if this is a fallback response
    const isGeneralKnowledgeFallback = answer.startsWith('[Note: No specific information was found');

    // Return the results
    return NextResponse.json({
      answer,
      usedFallback: isGeneralKnowledgeFallback,
      success: true
    });
  } catch (error) {
    console.error('Error in RAG query:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json(
      { error: `Failed to execute RAG query: ${errorMessage}` },
      { status: 500 }
    );
  }
} 