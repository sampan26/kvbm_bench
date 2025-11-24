import { NextRequest, NextResponse } from 'next/server';
import RAGService from '@/lib/rag';

/**
 * API endpoint for storing documents in the RAG system
 * POST /api/store-documents
 */
export async function POST(req: NextRequest) {
  try {
    // Parse request body
    const body = await req.json();
    const { documents, metadata } = body;

    if (!documents || !Array.isArray(documents) || documents.length === 0) {
      return NextResponse.json({ error: 'Documents array is required' }, { status: 400 });
    }

    // Validate that all documents are strings
    const isValid = documents.every(doc => typeof doc === 'string' && doc.trim().length > 0);
    if (!isValid) {
      return NextResponse.json({ 
        error: 'All documents must be non-empty strings' 
      }, { status: 400 });
    }

    // Initialize the RAG service
    const ragService = RAGService;
    await ragService.initialize();
    
    console.log(`Storing ${documents.length} documents in RAG system`);

    // Store the documents
    await ragService.storeDocuments(documents, metadata);

    // Return success
    return NextResponse.json({
      success: true,
      count: documents.length,
      message: `Successfully stored ${documents.length} documents`
    });
  } catch (error) {
    console.error('Error storing documents:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json(
      { error: `Failed to store documents: ${errorMessage}` },
      { status: 500 }
    );
  }
} 