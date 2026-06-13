#!/usr/bin/env python3
"""
Convert PDF pages to PNG images for visual review.
Uses pymupdf (fitz) for PDF rendering.
"""
import os
import sys

try:
    import fitz  # pymupdf
except ImportError:
    print("pymupdf not installed. Installing...")
    os.system(f"{sys.executable} -m pip install pymupdf")
    import fitz

def convert_pdf_to_png(pdf_path, output_dir, dpi=150):
    """
    Convert each page of a PDF to PNG image.
    
    Args:
        pdf_path: Path to input PDF file
        output_dir: Directory to save PNG files
        dpi: Resolution in DPI (default: 150)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open PDF
    doc = fitz.open(pdf_path)
    
    print(f"Converting {pdf_path} ({len(doc)} pages) to PNG at {dpi} DPI...")
    
    # Calculate zoom factor from DPI (72 DPI is base)
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    
    page_paths = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Render page to pixmap
        pix = page.get_pixmap(matrix=mat)
        
        # Save as PNG
        output_path = os.path.join(output_dir, f"page_{page_num + 1:02d}.png")
        pix.save(output_path)
        page_paths.append(output_path)
        print(f"  Page {page_num + 1}: {output_path}")
    
    doc.close()
    print(f"\nConverted {len(doc)} pages to {output_dir}/")
    return page_paths

if __name__ == "__main__":
    pdf_path = os.path.join(os.path.dirname(__file__), "paper.pdf")
    output_dir = os.path.join(os.path.dirname(__file__), "page_images")
    
    if not os.path.exists(pdf_path):
        print(f"ERROR: PDF not found: {pdf_path}")
        sys.exit(1)
    
    page_paths = convert_pdf_to_png(pdf_path, output_dir, dpi=150)
    print("\nDone! Page images saved to:", output_dir)
