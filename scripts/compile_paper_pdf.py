from __future__ import annotations
import os
import re
import fitz
from pathlib import Path
from markdown_pdf import MarkdownPdf, Section

def find_matching_brace(text: str, start_idx: int) -> int:
    """Finds the index of the matching closing curly brace for the one at start_idx."""
    count = 1
    for i in range(start_idx + 1, len(text)):
        if text[i] == '{':
            count += 1
        elif text[i] == '}':
            count -= 1
            if count == 0:
                return i
    return -1

def parse_fractions(text: str) -> str:
    """Parses LaTeX \\frac{num}{den} recursively by matching curly braces correctly into HTML slash fractions."""
    while True:
        match = re.search(r"\\frac\s*\{", text)
        if not match:
            break
        start_num = match.end() - 1  # index of '{'
        end_num = find_matching_brace(text, start_num)
        if end_num == -1:
            break
        num = text[start_num + 1:end_num]
        
        # Now find the denominator, which must immediately follow or have spaces
        rest = text[end_num + 1:]
        match_den = re.match(r"\s*\{", rest)
        if not match_den:
            break
        start_den = end_num + 1 + match_den.end() - 1
        end_den = find_matching_brace(text, start_den)
        if end_den == -1:
            break
        den = text[start_den + 1:end_den]
        
        # Replace the fraction block in the text
        fraction_str = text[match.start():end_den + 1]
        # Clean both numerator and denominator recursively
        num_clean = parse_fractions(num)
        den_clean = parse_fractions(den)
        
        fraction_html = f'<span class="math-fraction"><sup>{num_clean}</sup>⁄<sub>{den_clean}</sub></span>'
        text = text.replace(fraction_str, fraction_html)
    return text

def parse_sqrts(text: str) -> str:
    """Parses LaTeX \\sqrt{expr} recursively by matching curly braces correctly."""
    while True:
        match = re.search(r"\\sqrt\s*\{", text)
        if not match:
            break
        start_idx = match.end() - 1
        end_idx = find_matching_brace(text, start_idx)
        if end_idx == -1:
            break
        expr = text[start_idx + 1:end_idx]
        expr_clean = parse_sqrts(expr)
        sqrt_str = text[match.start():end_idx + 1]
        text = text.replace(sqrt_str, f'<span class="sqrt"><span class="symbol">√</span><span class="expr">{expr_clean}</span></span>')
    return text

def parse_bf_and_text(text: str) -> str:
    """Parses LaTeX \\mathbf{...} and \\text{...} recursively by matching curly braces correctly."""
    # Process \mathbf
    while True:
        match = re.search(r"\\mathbf\s*\{", text)
        if not match:
            break
        start_idx = match.end() - 1
        end_idx = find_matching_brace(text, start_idx)
        if end_idx == -1:
            break
        content = text[start_idx + 1:end_idx]
        content_clean = parse_bf_and_text(content)
        bf_str = text[match.start():end_idx + 1]
        text = text.replace(bf_str, f'<span class="math-bf">{content_clean}</span>')
        
    # Process \text
    while True:
        match = re.search(r"\\text\s*\{", text)
        if not match:
            break
        start_idx = match.end() - 1
        end_idx = find_matching_brace(text, start_idx)
        if end_idx == -1:
            break
        content = text[start_idx + 1:end_idx]
        content_clean = parse_bf_and_text(content)
        text_str = text[match.start():end_idx + 1]
        text = text.replace(text_str, f'<span class="math-text">{content_clean}</span>')
        
    # Process \mathrm
    while True:
        match = re.search(r"\\mathrm\s*\{", text)
        if not match:
            break
        start_idx = match.end() - 1
        end_idx = find_matching_brace(text, start_idx)
        if end_idx == -1:
            break
        content = text[start_idx + 1:end_idx]
        content_clean = parse_bf_and_text(content)
        mathrm_str = text[match.start():end_idx + 1]
        text = text.replace(mathrm_str, f'<span class="math-rm">{content_clean}</span>')
        
    return text

def parse_integrals(text: str) -> str:
    """Parses LaTeX \\int_{lower}^{upper} and \\int_lower^upper recursively into elegant inline-limit integral structures."""
    while True:
        # Match \int_{lower}^{upper}
        match = re.search(r"\\int\s*\_(?:\{([^{}]+)\}|([a-zA-Z0-9]))\s*\^(?:\{([^{}]+)\}|([a-zA-Z0-9]))", text)
        if match:
            lower = match.group(1) or match.group(2)
            upper = match.group(3) or match.group(4)
            int_html = f'<span class="math-integral">∫<sub>{lower}</sub><sup>{upper}</sup></span>'
            text = text.replace(match.group(0), int_html)
            continue
            
        # Match \int^{upper}_{lower}
        match = re.search(r"\\int\s*\^(?:\{([^{}]+)\}|([a-zA-Z0-9]))\s*\_(?:\{([^{}]+)\}|([a-zA-Z0-9]))", text)
        if match:
            upper = match.group(1) or match.group(2)
            lower = match.group(3) or match.group(4)
            int_html = f'<span class="math-integral">∫<sub>{lower}</sub><sup>{upper}</sup></span>'
            text = text.replace(match.group(0), int_html)
            continue
            
        # Match standalone \int
        match = re.search(r"\\int\b", text)
        if match:
            int_html = '<span class="math-integral">∫</span>'
            text = text.replace(match.group(0), int_html)
            continue
            
        break
    return text

def parse_sub_superscripts(text: str) -> str:
    """Parses LaTeX _{...} and ^{...} recursively into HTML sub and sup elements."""
    # Process _{...}
    while True:
        match = re.search(r"\_\s*\{", text)
        if not match:
            break
        start_idx = match.end() - 1
        end_idx = find_matching_brace(text, start_idx)
        if end_idx == -1:
            break
        content = text[start_idx + 1:end_idx]
        content_clean = parse_sub_superscripts(content)
        sub_str = text[match.start():end_idx + 1]
        text = text.replace(sub_str, f"<sub>{content_clean}</sub>")
        
    # Process ^{...}
    while True:
        match = re.search(r"\^\s*\{", text)
        if not match:
            break
        start_idx = match.end() - 1
        end_idx = find_matching_brace(text, start_idx)
        if end_idx == -1:
            break
        content = text[start_idx + 1:end_idx]
        content_clean = parse_sub_superscripts(content)
        sup_str = text[match.start():end_idx + 1]
        text = text.replace(sup_str, f"<sup>{content_clean}</sup>")
        
    # Process single char subscripts/superscripts (e.g. _l or ^2 or ^T)
    text = re.sub(r"\_([a-zA-Z0-9\+\-\=\(\)])", r"<sub>\1</sub>", text)
    text = re.sub(r"\^([a-zA-Z0-9\+\-\=\(\)])", r"<sup>\1</sup>", text)
    return text

def preprocess_math(text: str, output_dir: Path) -> str:
    """Converts LaTeX equations in Markdown to perfect, high-resolution transparent PNG images rendered via Matplotlib."""
    import hashlib
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # Create math images output directory
    img_dir = output_dir / "math_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure matplotlib for premium textbook Computer Modern serif math typography
    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    
    def get_equation_image(latex_str: str, is_block: bool) -> tuple[str, float]:
        eq_text = latex_str.strip()
        
        # Normalize double backslashes in cases or other environments
        eq_text = eq_text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Hash equation text to get a unique cached filename
        eq_hash = hashlib.md5(f"{eq_text}_{is_block}".encode('utf-8')).hexdigest()
        filename = f"eq_{eq_hash}.png"
        filepath = img_dir / filename
        
        if filepath.exists():
            try:
                with Image.open(filepath) as img:
                    return f"math_images/{filename}", img.height
            except Exception:
                pass # If file is corrupted, re-render it
                
        # Determine appropriate math canvas size and text size
        # We start with a large figure and crop tightly using bbox_inches='tight'
        fig = plt.figure(figsize=(10, 4))
        
        formula = eq_text
        if not formula.startswith('$'):
            formula = f"${formula}$"
            
        fontsize = 13 if is_block else 11.5
        
        # Render the equation
        fig.text(0.5, 0.5, formula, fontsize=fontsize, ha='center', va='center', color='#111111')
        
        # Save high-resolution transparent PNG
        fig.savefig(
            filepath,
            dpi=300,
            transparent=True,
            bbox_inches='tight',
            pad_inches=0.015 if is_block else 0.005
        )
        plt.close(fig)
        
        # Read the image height
        with Image.open(filepath) as img:
            return f"math_images/{filename}", img.height

    processed = text
    
    # 1. Replace block equations ($$ ... $$)
    def replace_block_math(match):
        latex = match.group(1).strip()
        try:
            img_path, img_height = get_equation_image(latex, is_block=True)
            pt_height = img_height * 72 / 300
            # Wrap in centered block container with correct height scaling
            return f'<div class="equation-block" style="text-align: center; margin: 16px 0; width: 100%;"><img src="{img_path}" style="height: {pt_height:.1f}pt; max-width: 95%; display: block; margin: 0 auto; vertical-align: middle;"></div>'
        except Exception as e:
            print(f"Warning: Failed to render block math '{latex}': {e}")
            # Fallback to text
            return f'<p class="equation-fallback" style="text-align: center; font-style: italic; margin: 15px 0;">{latex}</p>'
            
    processed = re.sub(r"\$\$(.*?)\$\$", replace_block_math, processed, flags=re.DOTALL)
    
    # 2. Replace inline equations ($ ... $)
    def replace_inline_math(match):
        latex = match.group(1).strip()
        # Avoid rendering empty dollars or spaces
        if not latex:
            return "$"
        try:
            img_path, img_height = get_equation_image(latex, is_block=False)
            pt_height = img_height * 72 / 300
            # Sits perfectly inline with a slight vertical-align offset
            # We use a negative vertical-align offset to align descenders beautifully with baseline
            return f'<img src="{img_path}" class="inline-math" style="height: {pt_height:.1f}pt; vertical-align: -18%; margin: 0 1px; display: inline-block;">'
        except Exception as e:
            print(f"Warning: Failed to render inline math '{latex}': {e}")
            return f'<span class="inline-math-fallback" style="font-style: italic;">{latex}</span>'
            
    processed = re.sub(r"\$([^$]+)\$", replace_inline_math, processed)
    return processed

def compile_paper():
    print("Starting CMF Research Paper PDF compilation with dynamic math processor...")
    
    workspace_dir = Path(__file__).resolve().parent.parent
    md_path = workspace_dir / "paper" / "continuous_meaning_field.md"
    pdf_path = workspace_dir / "paper" / "continuous_meaning_field.pdf"
    
    if not md_path.exists():
        print(f"Error: Paper markdown not found at {md_path}")
        return
        
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()
        
    # Preprocess the entire markdown content first, so all equations in the abstract and headings are rendered natively
    processed_content = preprocess_math(md_content, output_dir=workspace_dir / "paper")
    
    # Extract Title
    title_match = re.search(r"^# (.*?)$", processed_content, re.MULTILINE)
    title = title_match.group(1) if title_match else "Geodesics of Meaning"
    
    # Extract Author & Affiliation
    author_match = re.search(r"\*\*Aman Sachan\*\*\s*\n\*(.*?)\*\s*\n`(.*?)`", processed_content)
    if author_match:
        affiliation = author_match.group(1)
        email = author_match.group(2)
    else:
        affiliation = "Independent Researcher"
        email = "amansachan92905@gmail.com"
        
    # Extract Abstract text
    abstract_match = re.search(r"### Abstract\n(.*?)(?=\n##|$)", processed_content, re.DOTALL)
    abstract_text = abstract_match.group(1).strip() if abstract_match else ""
    
    # Clean up original headers from the body so we can render them via custom structured HTML
    body_content = processed_content
    body_content = re.sub(r"^# .*?$", "", body_content, flags=re.MULTILINE)
    body_content = re.sub(r"\*\*Aman Sachan\*\*.*?\n`.*?`", "", body_content, flags=re.DOTALL)
    body_content = re.sub(r"### Abstract.*?(\n##|$)", "\\1", body_content, flags=re.DOTALL)
    body_content = re.sub(r"^---\s*$", "", body_content, flags=re.MULTILINE) # remove top dividers
    
    # Reassemble with NeurIPS/ICML custom header/abstract block (CRITICAL: ZERO INDENTATION to avoid markdown code-block bug!)
    # We wrap the abstract_text directly inside abstract-box with blank lines so the Markdown parser
    # correctly parses all lists and inline formatting inside the abstract.
    html_header = f"""<div class="paper-header">
<h1 class="paper-title">{title}</h1>
<div class="author-block">
<span class="author-name">Aman Sachan</span>
</div>
<div class="abstract-box">
<div class="abstract-title">Abstract</div>

{abstract_text}

</div>
</div>"""
    
    final_markdown = html_header + "\n\n" + body_content.strip()
    
    # Premium LaTeX/NeurIPS Academic CSS stylesheet
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500&display=swap');
    
    @page {
        size: A4;
        margin: 15mm 15mm 15mm 15mm;
        background-color: #faf8f5 !important;
    }
    
    /* Strict Black & White, No-Background Stylesheet to prevent PyMuPDF/weasyprint container rectangle overlaps */
    
    /* Force all layout and textual elements to be transparent by default, excluding html/body */
    * {
        box-shadow: none !important;
        text-shadow: none !important;
    }
    
    div, p, span, blockquote, pre, code, table, tr, th, td, ul, ol, li, h1, h2, h3, h4, h5, h6 {
        background-color: transparent !important;
        background: transparent !important;
    }
    
    /* Explicitly define body background as soft off-white and text as warm off-black */
    html, body, p, span, blockquote, pre, code, ul, ol, li, h1, h2, h3, h4, h5, h6, table, tr, th, td {
        font-family: Helvetica, Arial, sans-serif !important;
    }
    
    p, li, blockquote, td, ol, ul, .abstract-box p {
        font-weight: 600 !important;
    }
    
    h1, h2, h3, h4, h5, h6, .paper-title, .abstract-title, .author-name {
        font-weight: 800 !important;
    }
    
    html, body {
        background-color: #faf8f5 !important;
        background: #faf8f5 !important;
        color: #111111 !important;
        line-height: 1.65;
        font-size: 10.5pt;
        margin: 0;
        padding: 0;
    }
    
    .paper-header {
        text-align: center;
        margin-bottom: 30px;
    }
    
    .paper-title {
        font-family: Helvetica, Arial, sans-serif !important;
        font-size: 22pt;
        font-weight: 700;
        color: #111111 !important;
        margin-top: 20px;
        margin-bottom: 15px;
        line-height: 1.25;
        text-align: center;
    }
    
    .author-block {
        margin-bottom: 25px;
        line-height: 1.5;
        text-align: center;
    }
    
    .author-name {
        font-family: Helvetica, Arial, sans-serif !important;
        font-size: 12.5pt;
        font-weight: 600;
        color: #111111 !important;
    }
    
    .author-affiliation {
        font-size: 10pt;
        color: #111111 !important;
        font-style: italic;
    }
    
    .author-email {
        font-size: 9pt;
        color: #111111 !important;
    }
    
    .abstract-box {
        max-width: 85%;
        margin: 25px auto 30px auto;
        padding-top: 15px;
        padding-bottom: 15px;
        border-top: 1.5px solid #111111 !important;
        border-bottom: 1.5px solid #111111 !important;
        text-align: justify;
    }
    
    .abstract-title {
        font-family: Helvetica, Arial, sans-serif !important;
        font-size: 10pt;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 8px;
        color: #111111 !important;
        text-align: center;
    }
    
    /* Ensure all elements inside the abstract inherit uniform academic font styles */
    .abstract-box p, .abstract-box li, .abstract-box ol, .abstract-box ul {
        font-family: Helvetica, Arial, sans-serif !important;
        font-size: 9.5pt !important;
        line-height: 1.5 !important;
        color: #111111 !important;
        text-align: justify;
    }
    
    .abstract-box li {
        margin-bottom: 0.4em;
    }
    
    h2, h3, h4 {
        font-family: Helvetica, Arial, sans-serif !important;
        color: #111111 !important;
        font-weight: 700;
        margin-top: 2em;
        margin-bottom: 0.6em;
    }
    
    h2 {
        font-size: 13.5pt;
        border-bottom: 1.5px solid #111111 !important;
        padding-bottom: 4px;
        margin-top: 2.2em;
    }
    
    h3 {
        font-size: 11.5pt;
        border-bottom: 1px solid #111111 !important;
        padding-bottom: 2px;
    }
    
    p {
        margin-top: 0;
        margin-bottom: 1.2em;
        text-align: justify;
        text-justify: inter-word;
        color: #111111 !important;
    }
    
    /* Mathematical Equation Styling */
    .equation, .inline-equation {
        font-family: 'Times New Roman', Times, serif !important;
    }
    
    .inline-equation {
        font-style: italic;
        padding: 0 1px;
    }
    
    .inline-equation .fraction {
        font-size: 85%;
    }
    
    .equation {
        text-align: center;
        margin: 20px 0;
        font-size: 11pt;
        line-height: 2.0;
        display: block;
        color: #111111 !important;
    }
    
    .math-fraction {
        display: inline !important;
        font-family: 'Times New Roman', Times, serif !important;
    }
    
    .math-fraction sup {
        font-weight: bold !important;
    }
    
    .math-fraction sub {
        font-weight: bold !important;
    }
    
    .sqrt {
        display: inline-flex;
        align-items: center;
        vertical-align: middle;
    }
    
    .sqrt .symbol {
        font-size: 13pt;
        margin-right: 1px;
        font-family: Helvetica, Arial, sans-serif !important;
    }
    
    .sqrt .expr {
        border-top: 1px solid #111111 !important;
        padding-top: 1px;
    }
    
    .math-integral {
        display: inline !important;
        font-family: 'Times New Roman', Times, serif !important;
        font-size: 15pt !important;
        vertical-align: middle !important;
    }
    
    .math-integral sub {
        font-weight: bold !important;
    }
    
    .math-integral sup {
        font-weight: bold !important;
    }
    
    .math-bf {
        font-weight: bold;
        font-style: normal;
    }
    
    .math-text, .math-rm {
        font-style: normal;
        font-family: Helvetica, Arial, sans-serif !important;
    }
    
    sup, sub {
        font-size: 7.5pt !important;
        line-height: 0 !important;
    }
    
    sup {
        vertical-align: super !important;
    }
    
    sub {
        vertical-align: sub !important;
    }
    
    code {
        font-family: Helvetica, Arial, sans-serif !important;
        font-size: 10.5pt !important;
        color: #111111 !important;
        padding: 0px 2px;
    }
    
    /* Code blocks use simple thin solid border, no colored background, reduced font size to prevent A4 paper overflow */
    pre {
        color: #111111 !important;
        border: 1px solid #111111 !important;
        padding: 10px;
        border-radius: 0px !important;
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        word-break: break-all !important;
        margin-top: 1em;
        margin-bottom: 1.5em;
        font-size: 7.5pt !important;
    }
    
    pre code {
        color: #111111 !important;
        font-size: 7.5pt !important;
        font-family: 'Courier New', Courier, monospace !important;
    }
    
    sub {
        vertical-align: sub !important;
        font-size: 7.5pt !important;
        line-height: 0 !important;
    }
    
    sup {
        vertical-align: super !important;
        font-size: 7.5pt !important;
        line-height: 0 !important;
    }
    
    /* Blockquotes use classic black left line with zero background color */
    blockquote {
        margin: 1.5em 0;
        padding: 10px 20px;
        border-left: 3px solid #111111 !important;
        color: #111111 !important;
        font-style: italic;
    }
    
    /* booktabs table: pure black rules, no vertical lines, no backgrounds */
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 2em 0;
        font-size: 9.5pt;
        border-top: 2px solid #111111 !important;
        border-bottom: 2px solid #111111 !important;
    }
    
    th, td {
        border: none !important;
        padding: 8px 12px;
        text-align: left;
        color: #111111 !important;
    }
    
    th {
        font-family: Helvetica, Arial, sans-serif !important;
        font-weight: 700;
        color: #111111 !important;
        border-bottom: 1.5px solid #111111 !important;
    }
    
    td {
        border-bottom: 1px solid #cbd5e1 !important;
    }
    
    tr:last-child td {
        border-bottom: none !important;
    }
    
    ul, ol {
        margin-top: 0;
        margin-bottom: 1.2em;
        padding-left: 20px;
    }
    
    li {
        margin-bottom: 0.5em;
        color: #111111 !important;
    }
    
    hr {
        border: 0;
        border-top: 1.5px solid #111111 !important;
        margin: 2.5em 0;
    }
    """
    
    pdf = MarkdownPdf(toc_level=0)
    pdf.add_section(Section(final_markdown, root=str(workspace_dir / "paper")), user_css=custom_css)
    
    temp_path = pdf_path.with_name("temp_compiled.pdf")
    print(f"Saving compiled paper PDF to temporary path {temp_path}...")
    pdf.save(str(temp_path))
    
    # Post-processing: Re-open the compiled PDF to paint the off-white background (#faf8f5) on the canvas behind the text
    doc = fitz.open(temp_path)
    bg_color = (250 / 255.0, 248 / 255.0, 245 / 255.0)
    for page in doc:
        page.wrap_contents()
        page.draw_rect(page.rect, color=bg_color, fill=bg_color, overlay=False)
    
    print(f"Saving final processed paper PDF to {pdf_path}...")
    doc.save(str(pdf_path), incremental=False)
    doc.close()
    
    if temp_path.exists():
        os.remove(temp_path)
    
    print("Success! CMF Research Paper PDF successfully compiled.")

if __name__ == "__main__":
    compile_paper()
