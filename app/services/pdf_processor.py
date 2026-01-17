from pdf2image import convert_from_path
from typing import List, Optional, Tuple, Dict, Any
import os
import tempfile
from PIL import Image
import numpy as np
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

# Try to import PyMuPDF with fallback
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    print("Warning: PyMuPDF not installed, some features will be limited")

class PDFProcessor:
    """PDF processing service"""
    
    def __init__(self):
        self.dpi = settings.OCR_DPI
        self.max_pages = 50  # Limit for processing
    
    async def process_pdf(self, pdf_path: str) -> Optional[List[np.ndarray]]:
        """
        Process PDF and convert to images
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of images as numpy arrays
        """
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            
            # Check file exists
            if not os.path.exists(pdf_path):
                logger.error(f"PDF file not found: {pdf_path}")
                return None
            
            # Extract metadata
            metadata = self._extract_pdf_metadata(pdf_path)
            logger.info(f"PDF metadata: {metadata}")
            
            # Convert PDF to images
            images = self._convert_pdf_to_images(pdf_path)
            
            if not images:
                logger.error("Failed to convert PDF to images")
                return None
            
            logger.info(f"Successfully converted {len(images)} pages to images")
            return images
            
        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            return None
    
    def _extract_pdf_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """Extract PDF metadata"""
        try:
            if not HAS_PYMUPDF:
                return {"pages": 0, "error": "PyMuPDF not available"}
                
            with fitz.open(pdf_path) as doc:
                metadata = {
                    "pages": len(doc),
                    "author": doc.metadata.get("author", ""),
                    "title": doc.metadata.get("title", ""),
                    "subject": doc.metadata.get("subject", ""),
                    "creation_date": doc.metadata.get("creationDate", ""),
                    "file_size": os.path.getsize(pdf_path)
                }
            return metadata
        except Exception as e:
            logger.warning(f"Failed to extract PDF metadata: {e}")
            return {"pages": 0, "error": str(e)}
    
    def _convert_pdf_to_images(self, pdf_path: str) -> List[np.ndarray]:
        """Convert PDF pages to images"""
        try:
            # Use pdf2image to convert PDF to images
            images_pil = convert_from_path(
                pdf_path,
                dpi=self.dpi,
                first_page=1,
                last_page=self.max_pages,
                fmt='jpeg',
                thread_count=4
            )
            
            # Convert PIL images to numpy arrays
            images_np = []
            for pil_img in images_pil:
                np_img = np.array(pil_img)
                
                # Convert RGBA to RGB if needed
                if np_img.shape[2] == 4:
                    np_img = np_img[:, :, :3]
                
                images_np.append(np_img)
            
            return images_np
            
        except Exception as e:
            logger.error(f"PDF to image conversion failed: {e}")
            
            # Fallback: Try PyMuPDF if available
            if HAS_PYMUPDF:
                try:
                    return self._convert_with_pymupdf(pdf_path)
                except Exception as e2:
                    logger.error(f"PyMuPDF fallback also failed: {e2}")
                    return []
            else:
                logger.error("PyMuPDF not available for fallback")
                return []
    
    def _convert_with_pymupdf(self, pdf_path: str) -> List[np.ndarray]:
        """Convert PDF using PyMuPDF as fallback"""
        images = []
        
        try:
            if not HAS_PYMUPDF:
                return []
                
            doc = fitz.open(pdf_path)
            
            for page_num in range(min(len(doc), self.max_pages)):
                page = doc[page_num]
                
                # Render page to image
                pix = page.get_pixmap(dpi=self.dpi)
                
                # Convert to numpy array
                img_data = np.frombuffer(pix.samples, dtype=np.uint8)
                img_data = img_data.reshape(pix.height, pix.width, pix.n)
                
                # Convert to RGB if needed
                if pix.n == 4:
                    # Remove alpha channel
                    img_data = img_data[:, :, :3]
                
                images.append(img_data)
            
            doc.close()
            return images
            
        except Exception as e:
            logger.error(f"PyMuPDF conversion failed: {e}")
            return []
    
    async def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract text directly from PDF (for searchable PDFs)
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with extracted text
        """
        try:
            if not HAS_PYMUPDF:
                return {
                    "success": False,
                    "error": "PyMuPDF not installed for text extraction"
                }
                
            logger.info(f"Extracting text from PDF: {pdf_path}")
            
            with fitz.open(pdf_path) as doc:
                text_by_page = []
                total_text = ""
                
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    page_text = page.get_text()
                    
                    text_by_page.append({
                        "page": page_num + 1,
                        "text": page_text,
                        "length": len(page_text)
                    })
                    
                    total_text += page_text + "\n\n"
            
            return {
                "success": True,
                "total_pages": len(text_by_page),
                "total_text_length": len(total_text),
                "text_by_page": text_by_page,
                "full_text": total_text
            }
            
        except Exception as e:
            logger.error(f"PDF text extraction failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def split_pdf_by_pages(self, 
                               pdf_path: str, 
                               output_dir: str,
                               pages_per_split: int = 10) -> List[str]:
        """
        Split PDF into multiple files
        
        Args:
            pdf_path: Input PDF path
            output_dir: Output directory
            pages_per_split: Pages per split file
            
        Returns:
            List of output file paths
        """
        try:
            if not HAS_PYMUPDF:
                logger.error("PyMuPDF required for PDF splitting")
                return []
                
            logger.info(f"Splitting PDF: {pdf_path}")
            
            os.makedirs(output_dir, exist_ok=True)
            output_files = []
            
            with fitz.open(pdf_path) as doc:
                total_pages = len(doc)
                
                for start in range(0, total_pages, pages_per_split):
                    end = min(start + pages_per_split, total_pages)
                    
                    # Create new PDF for this range
                    new_doc = fitz.open()
                    for page_num in range(start, end):
                        new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
                    
                    # Save output
                    output_path = os.path.join(
                        output_dir,
                        f"split_{start+1}_{end}.pdf"
                    )
                    new_doc.save(output_path)
                    new_doc.close()
                    
                    output_files.append(output_path)
            
            logger.info(f"Split PDF into {len(output_files)} files")
            return output_files
            
        except Exception as e:
            logger.error(f"PDF splitting failed: {e}")
            return []