import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import nltk
from nltk.tokenize import sent_tokenize
from langchain_core.documents import Document
import os
import logging
import hashlib
import concurrent.futures
from functools import lru_cache
import time
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("pdf_processor")

# Constants
MAX_PDF_SIZE_MB = 50
MAX_PAGE_COUNT = 500
OCR_TIMEOUT = 30  # seconds
DEFAULT_LANGUAGE = "eng"

class PDFProcessingError(Exception):
    """Custom exception for PDF processing errors"""
    pass

@lru_cache(maxsize=100)
def get_page_text_cached(pdf_hash: str, page_num: int) -> str:
    """Cached version of text extraction to avoid reprocessing"""
    # This is a placeholder - in production you'd use Redis/disk caching
    # Return cached text if available
    return ""  # Return empty if not in cache

def is_pdf_safe(pdf_path: str) -> bool:
    """Check if PDF is safe to process (size, page count, etc)"""
    try:
        file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
        if file_size_mb > MAX_PDF_SIZE_MB:
            logger.warning(f"PDF too large: {file_size_mb}MB > {MAX_PDF_SIZE_MB}MB")
            return False

        doc = fitz.open(pdf_path)
        if len(doc) > MAX_PAGE_COUNT:
            logger.warning(f"Too many pages: {len(doc)} > {MAX_PAGE_COUNT}")
            return False

        return True
    except Exception as e:
        logger.error(f"Error checking PDF safety: {e}")
        return False

def extract_text_from_page(doc: fitz.Document, page_num: int,
                          ocr_threshold: int = 20,
                          language: str = DEFAULT_LANGUAGE) -> List[Dict[str, str]]:
    """Extract text from a single page with improved error handling"""
    start_time = time.time()
    extracted_chunks = []

    try:
        page = doc[page_num]
        text = page.get_text().strip()

        # Use direct text if sufficient
        if len(text.split()) >= ocr_threshold:
            logger.debug(f"Using direct text extraction for page {page_num+1}")
            extracted_chunks.append({
                "text": text,
                "source": "text"
            })
            return extracted_chunks

        # Fall back to OCR
        logger.debug(f"Using OCR for page {page_num+1}")
        images = page.get_images(full=True)

        for img_idx, img in enumerate(images):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]

                with Image.open(io.BytesIO(image_bytes)).convert("RGB") as image:
                    # Set timeout for OCR to prevent hanging
                    ocr_text = pytesseract.image_to_string(
                        image,
                        lang=language,
                        timeout=OCR_TIMEOUT
                    ).strip()

                    if ocr_text:
                        extracted_chunks.append({
                            "text": ocr_text,
                            "source": "ocr"
                        })
            except Exception as e:
                logger.error(f"Error processing image {img_idx} on page {page_num+1}: {str(e)}")
                continue

        processing_time = time.time() - start_time
        logger.debug(f"Page {page_num+1} processed in {processing_time:.2f}s")
        return extracted_chunks

    except Exception as e:
        logger.error(f"Failed to process page {page_num+1}: {str(e)}")
        return []

def process_pdf_parallel(pdf_path: str, ocr_threshold: int = 20,
                        max_workers: int = 4,
                        language: str = DEFAULT_LANGUAGE) -> List[Dict[str, Any]]:
    """Process PDF with parallel execution for better performance"""
    if not is_pdf_safe(pdf_path):
        raise PDFProcessingError(f"PDF failed safety checks: {pdf_path}")

    all_chunks = []
    pdf_hash = hashlib.md5(open(pdf_path, 'rb').read()).hexdigest()

    try:
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        logger.info(f"Processing PDF with {page_count} pages: {pdf_path}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create processing jobs for each page
            future_to_page = {
                executor.submit(
                    extract_text_from_page, doc, page_num, ocr_threshold, language
                ): page_num
                for page_num in range(page_count)
            }

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    chunks = future.result()
                    for chunk in chunks:
                        all_chunks.append({
                            "page": page_num + 1,
                            "source": chunk["source"],
                            "text": chunk["text"]
                        })
                except Exception as e:
                    logger.error(f"Error processing page {page_num+1}: {str(e)}")

        return all_chunks
    except Exception as e:
        logger.error(f"Failed to process PDF {pdf_path}: {str(e)}")
        raise PDFProcessingError(f"PDF processing failed: {str(e)}")

def chunk_text(text: str, max_sentences: int = 5,
              min_chunk_chars: int = 100,
              max_chunk_chars: int = 4000) -> List[str]:
    """Enhanced text chunking with size constraints"""
    try:
        # Remove excessive whitespace
        text = " ".join(text.split())
        if not text:
            return []

        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            # Skip empty sentences
            if not sentence.strip():
                continue

            sentence_len = len(sentence)

            # If adding this sentence exceeds max length and we already have content,
            # finalize the current chunk and start a new one
            if current_chunk and (current_length + sentence_len > max_chunk_chars or
                                 len(current_chunk) >= max_sentences):
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0

            current_chunk.append(sentence)
            current_length += sentence_len

        # Add the last chunk if it exists and meets minimum size
        if current_chunk and current_length >= min_chunk_chars:
            chunks.append(" ".join(current_chunk))

        return chunks
    except Exception as e:
        logger.error(f"Error chunking text: {str(e)}")
        # Fallback to basic chunking
        if len(text) > max_chunk_chars:
            return [text[:max_chunk_chars]]
        return [text] if text.strip() else []

def convert_to_langchain_docs(
    pdf_path: str,
    max_sentences: int = 10,
    ocr_threshold: int = 20,
    max_workers: int = 4,
    language: str = DEFAULT_LANGUAGE
) -> List[Document]:
    """End-to-end processing to LangChain Documents with production features"""
    start_time = time.time()
    logger.info(f"Starting processing of {pdf_path}")

    try:
        # Process the PDF with parallel execution
        extracted = process_pdf_parallel(
            pdf_path,
            ocr_threshold=ocr_threshold,
            max_workers=max_workers,
            language=language
        )

        documents = []
        filename = os.path.basename(pdf_path)

        # Create document chunks
        for entry in extracted:
            page_chunks = chunk_text(
                entry["text"],
                max_sentences=max_sentences
            )

            for chunk_idx, chunk in enumerate(page_chunks):
                if not chunk.strip():
                    continue

                # Create unique ID for the chunk
                chunk_id = f"{filename}_p{entry['page']}_c{chunk_idx}"

                # Create LangChain Document
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": pdf_path,
                        "filename": filename,
                        "page": entry["page"],
                        "chunk_id": chunk_id,
                        "extraction_method": entry["source"],
                        "processing_time": time.time() - start_time
                    }
                )
                documents.append(doc)

        logger.info(f"Completed processing {pdf_path}: {len(documents)} chunks created")
        return documents


    except Exception as e:
        logger.error(f"Failed to process {pdf_path}: {str(e)}")
        raise PDFProcessingError(f"Document conversion failed: {str(e)}")
