import { DocumentsTable } from "@/components/documents-table";
import { DocumentProcessor } from "@/components/document-processor";

export default function DocumentDataPage() {
  return (
    <div className="container mx-auto py-8">
      <h1 className="text-2xl font-bold mb-6">Document Knowledge Graph Builder</h1>
      <p className="mb-4 text-muted-foreground">
        Process documents to extract knowledge triples and generate embeddings using LangChain.
      </p>
      
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
        <div className="xl:col-span-1">
          <DocumentProcessor className="sticky top-8" />
        </div>
        <div className="xl:col-span-2">
          <div className="bg-card rounded-lg border border-border shadow-sm overflow-hidden">
            <div className="p-5 border-b border-border">
              <h2 className="text-xl font-semibold">Documents</h2>
              <p className="text-sm text-muted-foreground mt-1">
                Manage your documents and generate embeddings directly from the table
              </p>
            </div>
            <DocumentsTable />
          </div>
        </div>
      </div>
    </div>
  );
} 