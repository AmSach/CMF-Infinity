import fitz
from pathlib import Path

def render_pages():
    workspace_dir = Path(__file__).resolve().parent.parent
    
    # 1. Render Textbook
    textbook_pdf = workspace_dir / "docs" / "cmf_masterclass_textbook.pdf"
    if textbook_pdf.exists():
        doc = fitz.open(textbook_pdf)
        print(f"Textbook opened successfully. Total pages: {len(doc)}")
        for i, page in enumerate(doc):
            image_path = workspace_dir / "docs" / f"cmf_masterclass_textbook_page_{i+1}.png"
            pix = page.get_pixmap(dpi=150)
            pix.save(str(image_path))
        print("Done rendering textbook pages.")
        
    # 2. Render Paper
    paper_pdf = workspace_dir / "paper" / "continuous_meaning_field.pdf"
    if paper_pdf.exists():
        doc = fitz.open(paper_pdf)
        print(f"Paper opened successfully. Total pages: {len(doc)}")
        for i, page in enumerate(doc):
            image_path = workspace_dir / "paper" / f"continuous_meaning_field_page_{i+1}.png"
            pix = page.get_pixmap(dpi=150)
            pix.save(str(image_path))
        print("Done rendering paper pages.")

if __name__ == "__main__":
    render_pages()
