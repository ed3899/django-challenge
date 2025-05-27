from core_processor.chromadb_manager import ChromaDBManager
import logging

logger = logging.getLogger(__name__)

def categorize_new_document_with_chromadb(new_document_text: str, n_results: int = 3):
    """
    Categorizes a new document by finding similar documents in ChromaDB.
    """
    if not ChromaDBManager._initialized or not chroma_manager.collection or not chroma_manager.embedding_function:
        logger.error("ChromaDB not initialized properly.")
        return None

    try:
        # 1. Generate embedding for the new text
        new_embedding = chroma_manager.embedding_function([new_document_text])[0]

        # 2. Query ChromaDB for similar documents
        results = chroma_manager.collection.query(
            query_embeddings=[new_embedding],
            n_results=n_results,
            include=['metadatas']
        )

        if results and results['ids'] and results['ids'][0]:
            # 3. Determine category based on the most similar results
            most_similar_types = [result['document_type'] for result in results['metadatas'][0]]
            logger.info(f"Top similar document types: {most_similar_types}")

            # Simple approach: take the type of the most similar document
            return most_similar_types[0] if most_similar_types else None
        else:
            logger.info("No similar documents found in ChromaDB.")
            return None

    except Exception as e:
        logger.error(f"Error categorizing new document with ChromaDB: {e}", exc_info=True)
        return None

# Example usage (assuming you have 'new_text' from a processed document):
# new_text = get_processed_text_of_new_document()
# category = categorize_new_document_with_chromadb(new_text)
# if category:
#     print(f"The new document is likely a: {category}")
# else:
#     print("Could not categorize the new document.")