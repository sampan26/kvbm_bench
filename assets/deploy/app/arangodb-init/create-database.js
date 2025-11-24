// ArangoDB initialization script to create the txt2kg database
// This script is executed automatically when the ArangoDB container starts

db._createDatabase("txt2kg");
console.log("Database 'txt2kg' created successfully!");

// Optional: Create collections needed by your application
// Replace with actual collections you need
/*
const db = require("@arangodb").db;
db._useDatabase("txt2kg");

if (!db._collection("entities")) {
  db._createDocumentCollection("entities");
  console.log("Collection 'entities' created");
}

if (!db._collection("relationships")) {
  db._createEdgeCollection("relationships");
  console.log("Collection 'relationships' created");
}
*/ 