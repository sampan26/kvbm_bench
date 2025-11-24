// Test script for LLMGraphTransformer - DEPRECATED
// xAI integration has been removed
// const { ChatXAI } = require('@langchain/xai');
const { LLMGraphTransformer } = require('@langchain/community/experimental/graph_transformers/llm');
const { Document } = require('langchain/document');

async function testGraphTransformer() {
  console.log('Testing LLMGraphTransformer...');
  
  try {
    // xAI integration has been removed - this script is deprecated
    console.error('This test script is deprecated - xAI integration has been removed');
    return;
    
    // const llm = new ChatXAI({
    //   model: "grok-2-latest",
    //   temperature: 0.1,
    //   apiKey: xaiApiKey
    // });
    
    // Initialize LLMGraphTransformer
    const graphTransformer = new LLMGraphTransformer({
      llm,
      allowedNodes: ["Person", "Organization", "Concept", "Location", "Event", "Product"],
      allowedRelationships: ["RELATED_TO", "PART_OF", "LOCATED_IN", "WORKS_AT", "CREATED", "BELONGS_TO", "HAS_PROPERTY"],
      nodeProperties: ["name", "type", "description"]
    });
    
    // Create a test document
    const text = "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity. He worked at the Patent Office in Bern.";
    const documents = [new Document({ pageContent: text })];
    
    // Convert to graph documents
    console.log('Converting document to graph...');
    const graphDocuments = await graphTransformer.convertToGraphDocuments(documents);
    
    // Display the result
    console.log('Graph Document:', JSON.stringify(graphDocuments, null, 2));
  } catch (error) {
    console.error('Error testing LLMGraphTransformer:', error);
  }
}

// Run the test
testGraphTransformer().catch(console.error); 