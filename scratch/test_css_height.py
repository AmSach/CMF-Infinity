import fitz
from markdown_pdf import MarkdownPdf, Section
from pathlib import Path

workspace_dir = Path("E:/CMF")
pdf_path = workspace_dir / "docs" / "test_css_height.pdf"

md_content = """# Test Title
This is some test content to see if the background color is uniform edge-to-edge with 100% height.

* Item 1
* Item 2
"""

custom_css = """
@page {
    size: A4;
    margin: 0;
}
html, body {
    height: 100%;
    background-color: #fdfbf7 !important;
    background: #fdfbf7 !important;
    color: #334155 !important;
    font-family: sans-serif;
    margin: 0;
    padding: 20mm 20mm 20mm 20mm;
    box-sizing: border-box;
}
"""

pdf = MarkdownPdf(toc_level=0)
pdf.add_section(
    Section(md_content, root=str(workspace_dir / "docs"), borders=(0, 0, 0, 0)),
    user_css=custom_css
)
pdf.save(str(pdf_path))

doc = fitz.open(pdf_path)
print(f"Total pages: {len(doc)}")
page = doc[0]
pix = page.get_pixmap(dpi=150)
pix.save("E:/CMF/docs/test_css_height_page_1.png")
doc.close()
print("Saved test_css_height_page_1.png")
