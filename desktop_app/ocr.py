"""
IsoCortex Desktop App — OCR Module
====================================
Extracts text from scanned PDFs and images using Tesseract OCR.
Used as a fallback when normal text extraction yields no results.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("IsoCortex.ocr")

# Supported image extensions for OCR
OCR_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif", ".webp"}

# Minimum image dimensions to bother with OCR (skip tiny icons/etc.)
MIN_IMAGE_DIMENSION = 50  # pixels


def check_tesseract_available() -> bool:
    """Check if Tesseract OCR is installed on the system."""
    import shutil
    return shutil.which("tesseract") is not None


def ocr_image(image_path: Path, lang: str = "eng") -> str:
    """Run OCR on a single image file.
    
    Args:
        image_path: Path to the image file.
        lang: Tesseract language code (default: 'eng' for English).
    
    Returns:
        Extracted text, or empty string on failure.
    """
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        logger.warning("OCR dependencies missing: %s", exc)
        return ""
    
    try:
        img = Image.open(str(image_path))
        
        # Skip tiny images (icons, thumbnails)
        if img.width < MIN_IMAGE_DIMENSION or img.height < MIN_IMAGE_DIMENSION:
            logger.debug("Skipping tiny image: %s (%dx%d)", image_path.name, img.width, img.height)
            return ""
        
        # Run OCR
        text = pytesseract.image_to_string(img, lang=lang).strip()
        
        if text:
            logger.info(
                "OCR extracted %d chars from %s",
                len(text), image_path.name,
            )
        
        return text
        
    except Exception as exc:
        logger.error("OCR failed for %s: %s", image_path.name, exc)
        return ""


def ocr_pdf(file_path: Path, lang: str = "eng", max_pages: int = 100) -> str:
    """Run OCR on a PDF that contains scanned images (no text layer).
    
    Extracts each page as an image and runs Tesseract on it.
    Only used as a fallback when normal PDF text extraction yields nothing.
    
    Args:
        file_path: Path to the PDF file.
        lang: Tesseract language code.
        max_pages: Maximum pages to process (for large scanned docs).
    
    Returns:
        Extracted text with <<<PAGE:N>>> markers, or empty string.
    """
    try:
        import fitz
        import pytesseract
        from PIL import Image
        import io
    except ImportError as exc:
        logger.warning("OCR PDF dependencies missing: %s", exc)
        return ""
    
    try:
        doc = fitz.open(str(file_path))
    except Exception as exc:
        logger.error("OCR PDF open failed for %s: %s", file_path.name, exc)
        return ""

    try:
        pages = []
        total = min(len(doc), max_pages)

        for i in range(total):
            try:
                page = doc[i]

                # Render page to image (higher DPI = better OCR)
                # 200 DPI is a good balance of speed and accuracy
                mat = fitz.Matrix(200 / 72, 200 / 72)
                pix = page.get_pixmap(matrix=mat)

                # Convert to PIL Image
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))

                # Run OCR
                text = pytesseract.image_to_string(img, lang=lang).strip()

                if text:
                    pages.append(f"<<<PAGE:{i + 1}>>>{text}")
                    logger.debug(
                        "OCR page %d/%d: %d chars from %s",
                        i + 1, total, len(text), file_path.name,
                    )

            except Exception as exc:
                logger.warning("OCR failed for page %d of %s: %s", i + 1, file_path.name, exc)
                continue

        result = "\n\n".join(pages)
        if result:
            logger.info(
                "OCR extracted %d chars from %d pages of %s",
                len(result), len(pages), file_path.name,
            )

        return result
    finally:
        doc.close()


def extract_with_ocr(file_path: Path, lang: str = "eng") -> str:
    """Determine the file type and run appropriate OCR.
    
    For images: direct OCR.
    For PDFs: OCR all pages as images.
    For other files: return empty string.
    
    Args:
        file_path: Path to the file.
        lang: Tesseract language code.
    
    Returns:
        Extracted text with page markers for PDFs, or empty string.
    """
    ext = file_path.suffix.lower()
    
    if ext in OCR_IMAGE_EXTENSIONS:
        return ocr_image(file_path, lang=lang)
    elif ext == ".pdf":
        return ocr_pdf(file_path, lang=lang)
    else:
        return ""