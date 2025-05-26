from PIL import Image
import pytesseract

def load_image_and_extract_text(image_path: str) -> str:
    """
    Extracts text from a JPG image using Tesseract OCR and returns the extracted text.

    Args:
        image_path (str): The path to the JPG image.

    Returns:
        str: The extracted text.
    """

    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text.strip()  # Return the extracted text

    except Exception as e:
        raise ValueError(f"Error processing image {image_path}: {e}")
