/**
 * Retrieval Augmented Generation (RAG) implementation using Pinecone and LangChain
 * This module provides a RetrievalQA chain using Pinecone as the vector store
 * Note: xAI integration has been removed - needs alternative LLM provider implementation
 */

import { ChatOpenAI } from "@langchain/openai";
import { Document } from "@langchain/core/documents";
import { RunnableSequence } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { PromptTemplate } from "@langchain/core/prompts";
import { PineconeService, DocumentSearchResult } from './pinecone';
import { EmbeddingsService } from './embeddings';

// Interface for records to store in Pinecone
interface PineconeRecord {
  id: string;
  values: number[];
  metadata?: Record<string, any>;
}

export class RAGService {
  private static instance: RAGService;
  private pineconeService: PineconeService;
  private embeddingsService: EmbeddingsService;
  private llm: ChatOpenAI | null = null;
  private initialized: boolean = false;
  private isInitializing: boolean = false;

  private constructor() {
    this.pineconeService = PineconeService.getInstance();
    this.embeddingsService = EmbeddingsService.getInstance();
  }

  /**
   * Get the singleton instance of RAGService
   */
  public static getInstance(): RAGService {
    if (!RAGService.instance) {
      RAGService.instance = new RAGService();
    }
    return RAGService.instance;
  }

  /**
   * Initialize the RAG service
   */
  public async initialize(): Promise<void> {
    if (this.initialized || this.isInitializing) {
      return;
    }

    this.isInitializing = true;
    
    try {
      console.log('Initializing RAG service...');

      // Initialize dependencies
      await this.pineconeService.initialize();
      await this.embeddingsService.initialize();

      // Initialize LLM - Try NVIDIA first, then fall back to error
      const nvidiaApiKey = process.env.NVIDIA_API_KEY;
      if (!nvidiaApiKey) {
        throw new Error('RAG service requires NVIDIA_API_KEY to be set in environment variables. xAI integration has been removed.');
      }
      
      // Note: This is a placeholder - NVIDIA LLM integration would need to be implemented
      // For now, we'll throw an error to indicate RAG service is not available
      throw new Error('RAG service is temporarily unavailable after xAI removal. Please implement alternative LLM provider.');

      this.initialized = true;
      console.log('RAG service initialized successfully');
    } catch (error) {
      console.error('Error initializing RAG service:', error);
      throw error;
    } finally {
      this.isInitializing = false;
    }
  }

  /**
   * Store documents in Pinecone for retrieval
   * @param documents Array of text documents to store
   * @param metadata Optional metadata for the documents
   */
  public async storeDocuments(
    documents: string[],
    metadata?: Record<string, any>[]
  ): Promise<void> {
    if (!this.initialized) {
      await this.initialize();
    }

    if (!documents || documents.length === 0) {
      console.warn('No documents provided to store');
      return;
    }

    console.log(`Storing ${documents.length} documents in Pinecone`);

    // Generate embeddings for documents
    const embeddings = await this.embeddingsService.encode(documents);

    // Prepare records for Pinecone
    const records: PineconeRecord[] = embeddings.map((embedding, i) => ({
      id: `doc_${Date.now()}_${i}`,
      values: embedding,
      metadata: {
        text: documents[i],
        timestamp: new Date().toISOString(),
        ...(metadata && metadata[i] ? metadata[i] : {})
      }
    }));

    // Store in Pinecone
    await this.pineconeService.upsertVectors(records);
    console.log(`Successfully stored ${records.length} document embeddings`);
  }

  /**
   * Perform question answering with document retrieval
   * @param query User query
   * @param topK Number of most similar documents to retrieve
   * @returns Answer generated from relevant context
   */
  public async retrievalQA(query: string, topK: number = 5): Promise<string> {
    if (!this.initialized) {
      await this.initialize();
    }

    if (!this.llm) {
      throw new Error('LLM not initialized');
    }

    // Generate embedding for query
    const queryEmbedding = (await this.embeddingsService.encode([query]))[0];

    // Retrieve similar documents from Pinecone
    const similarDocs = await this.pineconeService.findSimilarDocuments(queryEmbedding, topK);
    
    if (!similarDocs || similarDocs.length === 0) {
      console.log('No relevant documents found, falling back to LLM');
      
      // Define prompt template for standalone LLM response
      const fallbackPromptTemplate = PromptTemplate.fromTemplate(`
You are a helpful assistant answering questions based on your general knowledge.
Since no specific information was found in the knowledge base, please provide the best answer you can.

Question: {query}

Answer:
`);

      // Create fallback chain
      const fallbackChain = RunnableSequence.from([
        {
          query: () => query,
        },
        fallbackPromptTemplate,
        this.llm,
        new StringOutputParser(),
      ]);

      // Execute fallback chain
      const answer = await fallbackChain.invoke({});
      return `[Note: No specific information was found in the knowledge base. This answer is based on general knowledge.]\n\n${answer}`;
    }

    // Extract text from retrieved documents
    const context = similarDocs
      .map((doc: DocumentSearchResult) => doc.metadata?.text || '')
      .filter((text: string) => text.length > 0)
      .join('\n\n');

    // Define prompt template for QA
    const promptTemplate = PromptTemplate.fromTemplate(`
Answer the question based only on the following context:

Context:
{context}

Question: {query}

Answer:
`);

    // Create retrieval chain
    const retrievalChain = RunnableSequence.from([
      {
        context: () => context,
        query: () => query,
      },
      promptTemplate,
      this.llm,
      new StringOutputParser(),
    ]);

    // Execute chain
    const answer = await retrievalChain.invoke({});
    return answer;
  }

  /**
   * Perform similar document retrieval without QA
   * @param query Query text
   * @param topK Number of documents to retrieve
   * @returns Array of retrieved documents with similarity scores
   */
  public async retrieveSimilarDocuments(
    query: string,
    topK: number = 5
  ): Promise<Array<{ text: string; score: number; metadata?: any }>> {
    if (!this.initialized) {
      await this.initialize();
    }

    // Generate embedding for query
    const queryEmbedding = (await this.embeddingsService.encode([query]))[0];

    // Retrieve similar documents from Pinecone
    const similarDocs = await this.pineconeService.findSimilarDocuments(queryEmbedding, topK);
    
    return similarDocs.map((doc: DocumentSearchResult) => ({
      text: doc.metadata?.text || '',
      score: doc.score,
      metadata: doc.metadata
    }));
  }
}

export default RAGService.getInstance(); 