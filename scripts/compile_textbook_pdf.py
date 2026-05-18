from __future__ import annotations
import os
import re
from pathlib import Path
from markdown_pdf import MarkdownPdf, Section

def preprocess_math(text: str) -> str:
    """Converts LaTeX equations in Markdown to beautiful HTML/Unicode representations."""
    processed = text
    
    # Helper to clean up mathematical LaTeX syntax inside equations
    def clean_latex(latex: str) -> str:
        html = latex
        
        # Replace LaTeX bold/text/math tags
        html = re.sub(r"\\mathbf\{([a-zA-Z0-9_+=\-*|/()\[\]\s\\,\.]+)\}", r"<b>\1</b>", html)
        html = re.sub(r"\\text\{([a-zA-Z0-9_+=\-*|/()\[\]\s\\,\.]+)\}", r"\1", html)
        html = re.sub(r"\\mathrm\{([a-zA-Z0-9_+=\-*|/()\[\]\s\\,\.]+)\}", r"\1", html)
        
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
        
        # Square roots \sqrt{expression}
        html = re.sub(r"\\sqrt\{([^}]+)\}", r"&radic;<span style='border-top: 1px solid; padding-top: 1px;'>\1</span>", html)
        
        # Fractions \frac{numerator}{denominator}
        html = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"(\1) / (\2)", html)
        
        # Subscripts and Superscripts (braced first, then single characters)
        html = re.sub(r"\^\{([^}]+)\}", r"<sup>\1</sup>", html)
        html = re.sub(r"\_\{([^}]+)\}", r"<sub>\1</sub>", html)
        html = re.sub(r"\^([a-zA-Z0-9])", r"<sup>\1</sup>", html)
        html = re.sub(r"\_([a-zA-Z0-9])", r"<sub>\1</sub>", html)
        
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
    
    body {
        font-family: 'Lora', 'Georgia', serif;
        color: #1e293b;
        line-height: 1.6;
        font-size: 11pt;
        margin: 20mm 20mm 20mm 20mm;
    }
    
    h1, h2, h3, h4 {
        font-family: 'Outfit', sans-serif;
        color: #0f172a;
        font-weight: 600;
        margin-top: 1.5em;
        margin-bottom: 0.5em;
    }
    
    h1 {
        font-size: 28pt;
        text-align: center;
        margin-top: 100px;
        margin-bottom: 20px;
        color: #1e3a8a;
        font-weight: 800;
    }
    
    h2 {
        font-size: 18pt;
        border-bottom: 1.5px solid #e2e8f0;
        padding-bottom: 6px;
        margin-top: 2em;
        color: #1e3a8a;
        page-break-before: always;
    }
    
    h3 {
        font-size: 13pt;
        color: #2563eb;
        margin-top: 1.5em;
    }
    
    p {
        margin-top: 0;
        margin-bottom: 1em;
        text-align: justify;
    }
    
    .equation {
        text-align: center;
        font-style: italic;
        margin: 15px 0;
        font-size: 11.5pt;
        color: #0f172a;
    }
    
    code {
        font-family: 'Fira Code', 'Courier New', Courier, monospace;
        font-size: 9pt;
        background-color: #f1f5f9;
        color: #0f172a;
        padding: 2px 4px;
        border-radius: 4px;
    }
    
    pre {
        background-color: #0f172a;
        color: #f8fafc;
        padding: 12px;
        border-radius: 6px;
        overflow-x: auto;
        margin-bottom: 1.5em;
    }
    
    pre code {
        background-color: transparent;
        color: inherit;
        padding: 0;
        border-radius: 0;
        font-size: 8.5pt;
    }
    
    blockquote {
        margin: 1.5em 0;
        padding: 10px 20px;
        background-color: #f8fafc;
        border-left: 4px solid #3b82f6;
        color: #334155;
        font-style: italic;
        border-radius: 0 6px 6px 0;
    }
    
    table {
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 1.5em;
    }
    
    th, td {
        border: 1px solid #e2e8f0;
        padding: 8px 12px;
        text-align: left;
    }
    
    th {
        background-color: #f8fafc;
        font-weight: 600;
        color: #0f172a;
    }
    
    tr:nth-child(even) td {
        background-color: #fafafa;
    }
    
    ul, ol {
        margin-top: 0;
        margin-bottom: 1em;
        padding-left: 20px;
    }
    
    li {
        margin-bottom: 0.5em;
    }
    
    hr {
        border: 0;
        border-top: 1px solid #e2e8f0;
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
