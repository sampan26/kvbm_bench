import { GraphDBService, GraphDBType } from './graph-db-service';
import { Neo4jService } from './neo4j';
import { ArangoDBService } from './arangodb';

/**
 * Get the appropriate graph database service based on the graph database type.
 * This is useful for API routes that need direct access to a specific graph database.
 * 
 * @param graphDbType - The type of graph database to use
 */
export function getGraphDbService(graphDbType: GraphDBType = 'arangodb') {
  if (graphDbType === 'neo4j') {
    return Neo4jService.getInstance();
  } else if (graphDbType === 'arangodb') {
    return ArangoDBService.getInstance();
  } else {
    // Default to ArangoDB
    return ArangoDBService.getInstance();
  }
}

/**
 * Initialize the graph database directly (not using GraphDBService).
 * This is useful for API routes that need direct access to a specific graph database.
 * 
 * @param graphDbType - The type of graph database to use
 */
export async function initializeGraphDb(graphDbType: GraphDBType = 'arangodb'): Promise<void> {
  const service = getGraphDbService(graphDbType);
  
  if (graphDbType === 'neo4j') {
    // Get Neo4j credentials from environment
    const uri = process.env.NEO4J_URI;
    const username = process.env.NEO4J_USER || process.env.NEO4J_USERNAME;
    const password = process.env.NEO4J_PASSWORD;
    
    // Initialize Neo4j connection
    if (service instanceof Neo4jService) {
      service.initialize(uri, username, password);
    }
  } else if (graphDbType === 'arangodb') {
    // Get ArangoDB credentials from environment
    const url = process.env.ARANGODB_URL;
    const dbName = process.env.ARANGODB_DB;
    const username = process.env.ARANGODB_USER;
    const password = process.env.ARANGODB_PASSWORD;
    
    // Initialize ArangoDB connection
    if (service instanceof ArangoDBService) {
      await service.initialize(url, dbName, username, password);
    }
  }
} 