from __future__ import annotations
import os
import re
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
    """Parses LaTeX \\frac{num}{den} recursively by matching curly braces correctly."""
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
        text = text.replace(fraction_str, f"({num_clean}) / ({den_clean})")
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
        text = text.replace(sqrt_str, f"√({expr_clean})")
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
        text = text.replace(bf_str, f"<b>{content_clean}</b>")
        
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
        text = text.replace(text_str, content_clean)
        
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
        text = text.replace(mathrm_str, content_clean)
        
    return text

def parse_sub_superscripts(text: str) -> str:
    """Parses LaTeX _{...} and ^{...} recursively by matching curly braces correctly."""
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
        
    # Process single char subscripts/superscripts
    text = re.sub(r"\_([a-zA-Z0-9])", r"<sub>\1</sub>", text)
    text = re.sub(r"\^([a-zA-Z0-9])", r"<sup>\1</sup>", text)
    return text

def preprocess_math(text: str) -> str:
    """Converts LaTeX equations in Markdown to beautiful HTML/Unicode representations."""
    processed = text
    
    # Helper to clean up mathematical LaTeX syntax inside equations
    def clean_latex(latex: str) -> str:
        html = latex
        
        # Standard replacements
        html = html.replace(r"\mathbb{R}", "ℝ")
        html = html.replace(r"\in", "&isin;")
        html = html.replace(r"\dots", "…")
        html = html.replace(r"\cdot", "&middot;")
        html = html.replace(r"\times", "&times;")
        html = html.replace(r"\theta", "&theta;")
        html = html.replace(r"\tau", "&tau;")
        html = html.replace(r"\epsilon", "&epsilon;")
        html = html.replace(r"\sigma", "&sigma;")
        html = html.replace(r"\beta", "&beta;")
        html = html.replace(r"\sum", "&sum;")
        html = html.replace(r"\|", "|")
        html = html.replace(r"\leftarrow", "&larr;")
        html = html.replace(r"\rightarrow", "&rarr;")
        html = html.replace(r"\implies", "&rArr;")
        html = html.replace(r"\sim", "~")
        html = html.replace(r"\mathcal{N}", "𝒩")
        html = html.replace(r"\sin", "sin")
        html = html.replace(r"\partial", "&part;")
        html = html.replace(r"\eta", "&eta;")
        html = html.replace(r"\lambda", "&lambda;")
        html = html.replace(r"\mu", "&mu;")
        html = html.replace(r"\alpha", "&alpha;")
        html = html.replace(r"\gamma", "&gamma;")
        html = html.replace(r"\Delta", "&Delta;")
        html = html.replace(r"\nabla", "&nabla;")
        html = html.replace(r"\left", "")
        html = html.replace(r"\right", "")
        
        # Parse LaTeX elements using curly brace matching
        html = parse_bf_and_text(html)
        html = parse_sqrts(html)
        html = parse_fractions(html)
        html = parse_sub_superscripts(html)
        
        return html

    # 1. Replace block equations ($$ ... $$)
    def replace_block_math(match):
        latex = match.group(1).strip()
        cleaned = clean_latex(latex)
        return f'<p class="equation">{cleaned}</p>'
    
    processed = re.sub(r"\$\$(.*?)\$\$", replace_block_math, processed, flags=re.DOTALL)
    
    # 2. Replace inline equations ($ ... $)
    def replace_inline_math(match):
        latex = match.group(1).strip()
        cleaned = clean_latex(latex)
        return cleaned
        
    processed = re.sub(r"\$([^$]+)\$", replace_inline_math, processed)
    
    return processed

def compile_textbook():
    print("Starting Continuous Meaning Field Masterclass Textbook PDF compilation with dynamic math processor...")
    
    # Define paths
    workspace_dir = Path(__file__).resolve().parent.parent
    md_path = workspace_dir / "docs" / "cmf_masterclass_textbook.md"
    pdf_path = workspace_dir / "docs" / "cmf_masterclass_textbook.pdf"
    
    if not md_path.exists():
        print(f"Error: Markdown file not found at {md_path}")
        return
        
    # Read Markdown content
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()
        
    # Preprocess all equations inside the textbook dynamically
    processed_content = preprocess_math(md_content)
        
    # Standard premium academic/spaceflight CSS stylesheet
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;800&family=Fira+Code:wght@400;500&family=Lora:ital,wght@0,400;0,500;0,600;1,400&display=swap');
    
    /* Strict Black & White, No-Background Stylesheet to prevent PyMuPDF/weasyprint container rectangle overlaps */
    
    /* Force all layout and textual elements to be transparent by default */
    * {
        background-color: transparent !important;
        background: transparent !important;
        box-shadow: none !important;
        text-shadow: none !important;
    }
    
    /* Explicitly define body background as pure white and text as pure black */
    html, body {
        background-color: #ffffff !important;
        background: #ffffff !important;
        color: #000000 !important;
        font-family: 'Lora', 'Georgia', serif;
        line-height: 1.6;
        font-size: 11pt;
        margin: 20mm 20mm 20mm 20mm;
    }
    
    h1, h2, h3, h4 {
        font-family: 'Outfit', sans-serif;
        color: #000000 !important;
        font-weight: 600;
        margin-top: 1.5em;
        margin-bottom: 0.5em;
    }
    
    h1 {
        font-size: 28pt;
        text-align: center;
        margin-top: 100px;
        margin-bottom: 20px;
        color: #000000 !important;
        font-weight: 800;
    }
    
    h2 {
        font-size: 18pt;
        border-bottom: 1.5px solid #000000 !important;
        padding-bottom: 6px;
        margin-top: 2em;
        color: #000000 !important;
        page-break-before: always;
    }
    
    h3 {
        font-size: 13pt;
        color: #000000 !important;
        margin-top: 1.5em;
    }
    
    p {
        margin-top: 0;
        margin-bottom: 1em;
        text-align: justify;
        color: #000000 !important;
    }
    
    .equation {
        text-align: center;
        font-style: italic;
        margin: 15px 0;
        font-size: 11.5pt;
        color: #000000 !important;
    }
    
    code {
        font-family: 'Fira Code', 'Courier New', Courier, monospace;
        font-size: 9pt;
        color: #000000 !important;
        padding: 0px 2px;
    }
    
    /* Code blocks use simple thin solid border, no colored background */
    pre {
        color: #000000 !important;
        border: 1px solid #000000 !important;
        padding: 12px;
        border-radius: 0px !important;
        overflow-x: auto;
        margin-bottom: 1.5em;
    }
    
    pre code {
        color: #000000 !important;
        font-size: 8.5pt;
    }
    
    blockquote {
        margin: 1.5em 0;
        padding: 10px 20px;
        border-left: 3px solid #000000 !important;
        color: #000000 !important;
        font-style: italic;
    }
    
    table {
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 1.5em;
        border-top: 2px solid #000000 !important;
        border-bottom: 2px solid #000000 !important;
    }
    
    th, td {
        border: none !important;
        padding: 8px 12px;
        text-align: left;
        color: #000000 !important;
    }
    
    th {
        font-weight: 700;
        color: #000000 !important;
        border-bottom: 1.5px solid #000000 !important;
    }
    
    td {
        border-bottom: 1px solid #cbd5e1 !important;
    }
    
    tr:last-child td {
        border-bottom: none !important;
    }
    
    ul, ol {
        margin-top: 0;
        margin-bottom: 1em;
        padding-left: 20px;
    }
    
    li {
        margin-bottom: 0.5em;
        color: #000000 !important;
    }
    
    hr {
        border: 0;
        border-top: 1.5px solid #000000 !important;
        margin: 2em 0;
    }
    """
    
    # Initialize PDF generator
    pdf = MarkdownPdf(toc_level=2)
    
    # Add textbook section with CSS
    pdf.add_section(Section(processed_content), user_css=custom_css)
    
    # Save the output PDF
    print(f"Saving compiled PDF to {pdf_path}...")
    pdf.save(str(pdf_path))
    print("Success! CMF Masterclass Textbook PDF successfully compiled and updated.")

if __name__ == "__main__":
    compile_textbook()
