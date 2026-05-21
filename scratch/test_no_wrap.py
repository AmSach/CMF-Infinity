import fitz
from markdown_pdf import MarkdownPdf, Section
from pathlib import Path

workspace_dir = Path("E:/CMF")
temp_path = workspace_dir / "docs" / "test_temp.pdf"
pdf_path = workspace_dir / "docs" / "test_no_wrap.pdf"

md_content = """# Test Title
This is some test content to see if drawing background without wrap_contents works.

* Item 1
* Item 2
"""

custom_css = """
html, body {
    color: #334155 !important;
    font-family: sans-serif;
    line-height: 1.7;
}
"""

pdf = MarkdownPdf(toc_level=0)
pdf.add_section(
    Section(md_content, root=str(workspace_dir / "docs")),
    user_css=custom_css
)
pdf.save(str(temp_path))

# Post-processing WITHOUT wrap_contents
doc = fitz.open(temp_path)
bg_color = (253/255, 251/255, 247/255)
for page in doc:
    # Do NOT call page.wrap_contents()
    page.draw_rect(page.rect, color=bg_color, fill=bg_color, overlay=False)
doc.save(str(pdf_path))
doc.close()

# Render page 1 to check background and text
doc = fitz.open(pdf_path)
print(f"Total pages: {len(doc)}")
page = doc[0]
pix = page.get_pixmap(dpi=150)
pix.save("E:/CMF/docs/test_no_wrap_page_1.png")
doc.close()
print("Saved test_no_wrap_page_1.png")
import os
if temp_path.exists():
    os.remove(temp_path)
