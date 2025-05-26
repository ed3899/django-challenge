from PIL import Image
import pytesseract

def load_image_and_extract_text(image_path: str) -> str:
    """
    Extracts text from a JPG image using Tesseract OCR and returns a LangChain Document.

    Args:
        image_path (str): The path to the JPG image.

    Returns:
        Document: A LangChain Document containing the extracted text.
    """

    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text.strip()  # Return the extracted text
        # return Document(page_content=text, metadata={"source": image_path})  # You can add more metadata
    except Exception as e:
        raise ValueError(f"Error processing image {image_path}: {e}")
