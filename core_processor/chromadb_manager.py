import chromadb
from chromadb.utils import embedding_functions

class ChromaDBManager:
    """
    Manages interactions with the ChromaDB vector database for document storage and retrieval.
    """

    def __init__(self, host: str = "localhost", port: int = 8000, collection_name: str = "documents"):
        """
        Initializes the ChromaDB client and gets or creates a collection.

        Args:
            host (str): The host address of the ChromaDB server.
            port (int): The port of the ChromaDB server.
            collection_name (str): The name of the collection to use.
        """
        try:
            # Initialize the ChromaDB client
            # For a persistent client, you would use:
            # self.client = chromadb.PersistentClient(path="/path/to/your/db")
            # For a client connecting to a running server:
            self.client = chromadb.HttpClient(host=host, port=port)
            print(f"Connected to ChromaDB at {host}:{port}")

            # Get or create the collection
            # Using a default embedding function for now. In a real application,
            # this would be integrated with the chosen LLM or a dedicated embedding model.
            # For example, using a Sentence Transformers embedding function:
            # self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            #     model_name="all-MiniLM-L6-v2"
            # )
            # For simplicity, let's use a mock embedding function or a basic one if available
            # that doesn't require external downloads for this initial setup.
            # ChromaDB's default is often satisfactory for initial testing.
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                # embedding_function=self.embedding_function # Uncomment and configure if needed
            )
            print(f"ChromaDB collection '{collection_name}' ready.")

        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
            self.client = None
            self.collection = None

    def add_document(self, doc_id: str, document: str, metadata: dict = None):
        """
        Adds a single document to the ChromaDB collection.

        Args:
            doc_id (str): A unique ID for the document.
            document (str): The text content of the document.
            metadata (dict, optional): A dictionary of metadata associated with the document.
                                       Defaults to None.
        """
        if not self.collection:
            print("ChromaDB not initialized. Cannot add document.")
            return

        try:
            self.collection.add(
                documents=[document],
                metadatas=[metadata if metadata else {}],
                ids=[doc_id]
            )
            print(f"Document '{doc_id}' added to ChromaDB.")
        except Exception as e:
            print(f"Error adding document '{doc_id}': {e}")

    def upsert_document(self, doc_id: str, document: str, metadata: dict = None):
        """
        Upserts (updates if exists, inserts if not) a single document to the ChromaDB collection.

        Args:
            doc_id (str): A unique ID for the document.
            document (str): The text content of the document.
            metadata (dict, optional): A dictionary of metadata associated with the document.
                                       Defaults to None.
        """
        if not self.collection:
            print("ChromaDB not initialized. Cannot upsert document.")
            return

        try:
            self.collection.upsert(
                documents=[document],
                metadatas=[metadata if metadata else {}],
                ids=[doc_id]
            )
            print(f"Document '{doc_id}' upserted in ChromaDB.")
        except Exception as e:
            print(f"Error upserting document '{doc_id}': {e}")

    def query_documents(self, query_texts: list[str], n_results: int = 5, where_clause: dict = None):
        """
        Queries the ChromaDB collection for similar documents.

        Args:
            query_texts (list[str]): A list of text strings to query for.
            n_results (int): The number of similar results to return.
            where_clause (dict, optional): A dictionary for metadata filtering. Defaults to None.

        Returns:
            dict: A dictionary containing the query results.
        """
        if not self.collection:
            print("ChromaDB not initialized. Cannot query documents.")
            return {}

        try:
            results = self.collection.query(
                query_texts=query_texts,
                n_results=n_results,
                where=where_clause
            )
            print(f"Query for '{query_texts}' completed.")
            return results
        except Exception as e:
            print(f"Error querying documents: {e}")
            return {}

    def get_document_by_id(self, doc_id: str):
        """
        Retrieves a document by its ID.

        Args:
            doc_id (str): The ID of the document to retrieve.

        Returns:
            dict: A dictionary containing the document details, or None if not found.
        """
        if not self.collection:
            print("ChromaDB not initialized. Cannot get document by ID.")
            return None

        try:
            result = self.collection.get(ids=[doc_id])
            if result and result['ids']:
                print(f"Document '{doc_id}' retrieved.")
                return {
                    "id": result['ids'][0],
                    "document": result['documents'][0] if result['documents'] else None,
                    "metadata": result['metadatas'][0] if result['metadatas'] else None
                }
            else:
                print(f"Document '{doc_id}' not found.")
                return None
        except Exception as e:
            print(f"Error retrieving document '{doc_id}': {e}")
            return None

    def delete_document(self, doc_id: str):
        """
        Deletes a document from the ChromaDB collection by its ID.

        Args:
            doc_id (str): The ID of the document to delete.
        """
        if not self.collection:
            print("ChromaDB not initialized. Cannot delete document.")
            return

        try:
            self.collection.delete(ids=[doc_id])
            print(f"Document '{doc_id}' deleted from ChromaDB.")
        except Exception as e:
            print(f"Error deleting document '{doc_id}': {e}")

# Example Usage (for testing purposes, can be removed or wrapped in main guard)
if __name__ == "__main__":
    # Ensure a ChromaDB server is running at localhost:8000 or adjust host/port
    # You can run a ChromaDB server using:
    # `chroma run --path /path/to/your/db` or `docker run -p 8000:8000 chromadb/chroma`

    db_manager = ChromaDBManager(host="localhost", port=8000, collection_name="my_documents")

    # Add some documents
    db_manager.add_document(
        doc_id="invoice_123",
        document="This is an invoice for services rendered on 2023-01-15 for $150.00.",
        metadata={"document_type": "invoice", "date": "2023-01-15"}
    )
    db_manager.add_document(
        doc_id="assignment_a1",
        document="Please complete the Python programming assignment by next Friday.",
        metadata={"document_type": "assignment", "due_date": "next Friday"}
    )
    db_manager.add_document(
        doc_id="form_hr_001",
        document="Employee onboarding form. Please fill out your personal details.",
        metadata={"document_type": "form", "department": "HR"}
    )

    # Query documents
    print("\n--- Querying for invoices ---")
    invoice_results = db_manager.query_documents(
        query_texts=["invoice details"],
        n_results=2,
        where_clause={"document_type": "invoice"}
    )
    print(invoice_results)

    print("\n--- Querying for assignments ---")
    assignment_results = db_manager.query_documents(
        query_texts=["programming tasks"],
        n_results=1,
        where_clause={"document_type": "assignment"}
    )
    print(assignment_results)

    # Get a document by ID
    print("\n--- Getting document by ID ---")
    doc_retrieved = db_manager.get_document_by_id("invoice_123")
    print(doc_retrieved)

    # Upsert a document (update existing)
    print("\n--- Upserting document ---")
    db_manager.upsert_document(
        doc_id="invoice_123",
        document="Updated invoice for services rendered on 2023-01-15 for $175.00.",
        metadata={"document_type": "invoice", "date": "2023-01-15", "amount": 175.00}
    )
    doc_updated = db_manager.get_document_by_id("invoice_123")
    print(doc_updated)

    # Delete a document
    print("\n--- Deleting document ---")
    db_manager.delete_document("form_hr_001")
    doc_deleted = db_manager.get_document_by_id("form_hr_001")
    print(doc_deleted) # Should show None or not found
