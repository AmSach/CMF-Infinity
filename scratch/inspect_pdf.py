import fitz

pdf_path = "E:/CMF/docs/cmf_masterclass_textbook.pdf"
doc = fitz.open(pdf_path)
print(f"Checking {pdf_path}:")
for i, page in enumerate(doc):
    print(f"Page {i+1}: rect={page.rect}, rotation={page.rotation}")
    images = page.get_images()
    if images:
        print(f"  Images on page {i+1}:")
        for img in images:
            print(f"    xref={img[0]}, width={img[2]}, height={img[3]}, size={img[4]}")
doc.close()
