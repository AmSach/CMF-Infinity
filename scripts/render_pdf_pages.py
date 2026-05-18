import fitz
from pathlib import Path

def render_pages():
    workspace_dir = Path(__file__).resolve().parent.parent
    pdf_path = workspace_dir / "paper" / "continuous_meaning_field.pdf"
    
    if not pdf_path.exists():
        print(f"Error: PDF not found at {pdf_path}")
        return
        
    doc = fitz.open(pdf_path)
    print(f"PDF opened successfully. Total pages: {len(doc)}")
    
    for i, page in enumerate(doc):
        image_path = workspace_dir / "paper" / f"continuous_meaning_field_page_{i+1}.png"
        print(f"Rendering page {i+1} to {image_path}...")
        pix = page.get_pixmap(dpi=150)
        pix.save(str(image_path))
        
    print("Done rendering pages as images.")

if __name__ == "__main__":
    render_pages()
