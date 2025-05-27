import chromadb
from chromadb.utils import embedding_functions
from django.conf import settings
import logging
import os
import uuid # For generating unique IDs for documents

# Import our new OCR and text cleaning utilities
from core_processor.ocr_manager import load_image_and_extract_text
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


logger = logging.getLogger(__name__)

class ChromaDBManager:
    """
    Manages interactions with the ChromaDB vector database.
    Handles connection, collection creation, upserting documents, and querying.
    Implements a singleton pattern for use across the Django application.
    """
    _instance = None # Singleton instance

    def __new__(cls):
        """
        Implements a singleton pattern to ensure only one instance of ChromaDBManager exists.
        This prevents multiple connections to ChromaDB.
        """
        if cls._instance is None:
            cls._instance = super(ChromaDBManager, cls).__new__(cls)
            cls._instance._initialized = False # Flag to ensure initialization runs only once
        return cls._instance

    def __init__(self):
        """
        Initializes the ChromaDB client and collection using Django settings.
        Ensures initialization happens only once for the singleton.
        """
        if self._initialized:
            return

        self.client = None
        self.collection = None
        self.embedding_function = None

        try:
            # Initialize ChromaDB client
            self.client = chromadb.HttpClient(
                host=settings.CHROMADB_HOST,
                port=settings.CHROMADB_PORT,
                headers={"Authorization": f"Bearer {settings.CHROMADB_AUTH_TOKEN}"}
            )
            logger.info(f"Successfully connected to ChromaDB at {settings.CHROMADB_URL}")

            # Define the embedding function.
            # This is crucial for ChromaDB to convert text into embeddings for similarity search.
            # We'll use a Sentence Transformer model as a robust general-purpose embedding.
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2" # A widely used and efficient embedding model
            )

            # Get or create the collection with the specified embedding function
            self.collection = self.client.get_or_create_collection(
                name=settings.CHROMADB_COLLECTION_NAME,
                embedding_function=self.embedding_function
            )
            logger.info(f"ChromaDB collection '{settings.CHROMADB_COLLECTION_NAME}' ready with embedding function.")
            self._initialized = True # Mark as initialized

            # Attempt to pre-load the embedding model if it hasn't been
            try:
                # This call forces the model download if it's the first run
                self.embedding_function(["test sentence"])
                logger.info("Embedding model 'all-MiniLM-L6-v2' pre-loaded successfully.")
            except Exception as e:
                logger.warning(f"Could not pre-load SentenceTransformer model: {e}")


        except Exception as e:
            logger.error(f"Failed to connect or initialize ChromaDB: {e}", exc_info=True)
            self.client = None
            self.collection = None
            self._initialized = False # Ensure flag is false if initialization fails

    def add_document(self, doc_id: str, document: str, metadata: dict = None):
        """
        Adds a single document to the ChromaDB collection.
        Uses the collection's default embedding function to generate embeddings.
        """
        if not self._initialized or not self.collection:
            logger.error("ChromaDBManager not initialized. Cannot add document.")
            return False

        try:
            self.collection.add(
                documents=[document],
                metadatas=[metadata if metadata else {}],
                ids=[doc_id]
            )
            logger.info(f"Document '{doc_id}' added to ChromaDB.")
            return True
        except Exception as e:
            logger.error(f"Error adding document '{doc_id}': {e}", exc_info=True)
            return False

    def upsert_document(self, doc_id: str, document: str, metadata: dict = None):
        """
        Upserts (updates if exists, inserts if not) a single document to the ChromaDB collection.
        Uses the collection's default embedding function to generate embeddings.
        """
        if not self._initialized or not self.collection:
            logger.error("ChromaDBManager not initialized. Cannot upsert document.")
            return False

        try:
            self.collection.upsert(
                documents=[document],
                metadatas=[metadata if metadata else {}],
                ids=[doc_id]
            )
            logger.info(f"Document '{doc_id}' upserted in ChromaDB.")
            return True
        except Exception as e:
            logger.error(f"Error upserting document '{doc_id}': {e}", exc_info=True)
            return False

    def query_documents(self, query_texts: list[str], n_results: int = 5, where_clause: dict = None):
        """
        Queries the ChromaDB collection for similar documents.
        """
        if not self._initialized or not self.collection:
            logger.error("ChromaDBManager not initialized. Cannot query documents.")
            return {}

        try:
            results = self.collection.query(
                query_texts=query_texts,
                n_results=n_results,
                where=where_clause,
                include=['metadatas', 'documents', 'distances'] # Include metadata for categorization
            )
            logger.info(f"Query for '{query_texts[0]}' completed. Found {len(results.get('ids', [[]])[0])} results.")
            return results
        except Exception as e:
            logger.error(f"Error querying documents: {e}", exc_info=True)
            return {}

    def get_document_by_id(self, doc_id: str):
        """
        Retrieves a document by its ID.
        """
        if not self._initialized or not self.collection:
            logger.error("ChromaDBManager not initialized. Cannot get document by ID.")
            return None

        try:
            result = self.collection.get(ids=[doc_id], include=['documents', 'metadatas'])
            if result and result['ids']:
                logger.info(f"Document '{doc_id}' retrieved.")
                return {
                    "id": result['ids'][0],
                    "document": result['documents'][0] if result['documents'] else None,
                    "metadata": result['metadatas'][0] if result['metadatas'] else None
                }
            else:
                logger.info(f"Document '{doc_id}' not found.")
                return None
        except Exception as e:
            logger.error(f"Error retrieving document '{doc_id}': {e}", exc_info=True)
            return None

    def delete_document(self, doc_id: str):
        """
        Deletes a document from the ChromaDB collection by its ID.
        """
        if not self._initialized or not self.collection:
            logger.error("ChromaDBManager not initialized. Cannot delete document.")
            return False

        try:
            self.collection.delete(ids=[doc_id])
            logger.info(f"Document '{doc_id}' deleted from ChromaDB.")
            return True
        except Exception as e:
            logger.error(f"Error deleting document '{doc_id}': {e}", exc_info=True)
            return False

    def categorize_new_document_with_chromadb(self, new_document_text: str, n_results: int = 3):
        """
        Categorizes a new document by finding similar documents in ChromaDB.
        Assumes the existing documents in ChromaDB have 'document_type' in their metadata.

        Args:
            new_document_text (str): The cleaned text of the new document to categorize.
            n_results (int): The number of similar documents to retrieve for categorization.

        Returns:
            str or None: The predicted document type, or None if categorization fails or no
                         similar documents are found.
        """
        if not self._initialized or not self.collection or not self.embedding_function:
            logger.error("ChromaDBManager not initialized or missing embedding function. Cannot categorize document.")
            return None

        try:
            # Query ChromaDB for similar documents using the new document's text
            # The collection's embedding function will convert new_document_text to an embedding
            results = self.query_documents(query_texts=[new_document_text], n_results=n_results)

            if results and results.get('metadatas') and results['metadatas'][0]:
                # Extract document types from the metadata of the similar results
                similar_document_types = [
                    meta.get('document_type')
                    for meta in results['metadatas'][0]
                    if meta and 'document_type' in meta
                ]

                if similar_document_types:
                    # Simple categorization: take the type of the most similar document
                    # For more robustness, one could implement a majority vote or weighted vote
                    # based on distances.
                    predicted_type = similar_document_types[0]
                    logger.info(f"Categorized new document as: '{predicted_type}' based on similarity.")
                    return predicted_type
                else:
                    logger.info("No 'document_type' found in metadata of similar documents.")
                    return None
            else:
                logger.info("No similar documents found in ChromaDB for categorization.")
                return None

        except Exception as e:
            logger.error(f"Error categorizing new document with ChromaDB: {e}", exc_info=True)
            return None

    def load_initial_dataset(self, dataset_path: str):
        """
        Recursively analyzes a directory structure, processes each image/PDF using OCR,
        cleans the text, infers the category from the directory name, and stores it in ChromaDB.

        Assumes the directory structure is like:
        dataset_path/
        ├── categoryA/
        │   ├── file1.jpg
        │   └── file2.pdf
        └── categoryB/
            ├── file3.png
            └── file4.jpg

        Args:
            dataset_path (str): The root path of the dataset directory.
        """
        if not self._initialized or not self.collection:
            logger.error("ChromaDBManager not initialized. Cannot load initial dataset.")
            return

        if not os.path.isdir(dataset_path):
            logger.error(f"Dataset path '{dataset_path}' is not a valid directory.")
            return

        print(f"Starting to load initial dataset from: {dataset_path}")
        processed_count = 0
        skipped_count = 0

        for root, dirs, files in os.walk(dataset_path):
            # Infer category from the immediate parent directory name
            # If at the root of the dataset_path, category might be 'unknown' or skipped
            relative_path = os.path.relpath(root, dataset_path)
            if relative_path == '.': # This is the root directory itself
                category = "unclassified" # Or you might skip files directly in the root
            else:
                category = os.path.basename(root) # The immediate parent directory name is the category

            for file_name in files:
                file_path = os.path.join(root, file_name)
                doc_id = str(uuid.uuid4()) # Generate a unique ID for each document

                print(f"Processing file: {file_path} (Category: {category})")

                try:
                    # 1. Extract text using OCR
                    raw_text = load_image_and_extract_text(file_path)
                    if not raw_text:
                        print(f"No text extracted from {file_name}. Skipping.")
                        skipped_count += 1
                        continue

                    # Depending on the level of complexity an agentic workflow could be introduced here
                    prompt_template = ChatPromptTemplate.from_messages(
                        [
                            (
                                "system",
                                "You talk like a pirate. Answer all questions to the best of your ability.",
                            ),
                            MessagesPlaceholder(variable_name="messages"),
                        ]
                    )

                    # 2. Prepare metadata
                    metadata = {
                        "file_name": file_name,
                        "original_path": file_path,
                        "document_type": category, # Use the inferred category as the document type
                        "source": "initial_dataset_load"
                    }

                    # 3. Store in ChromaDB
                    success = self.upsert_document(doc_id, raw_text, metadata)
                    if success:
                        processed_count += 1
                    else:
                        skipped_count += 1 # Upsert failed for some reason
                        print(f"Failed to upsert document {file_name} into ChromaDB.")

                except Exception as e:
                    print(f"Error processing file {file_path}: {e}", exc_info=True)
                    skipped_count += 1

        print(f"Finished loading initial dataset. Processed: {processed_count} documents, Skipped: {skipped_count} documents.")


# Global instance for easy access throughout the Django app
chroma_manager = ChromaDBManager()
