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
        
        # Replace LaTeX bold/text tags
        html = re.sub(r"\\mathbf\{([a-zA-Z0-9_+=\-*|/()\[\]\s]+)\}", r"<b>\1</b>", html)
        html = re.sub(r"\\text\{([a-zA-Z0-9_+=\-*|/()\[\]\s]+)\}", r"\1", html)
        
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
    processed_content = preprocess_math(md_content)
    
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
    html_header = f"""<div class="paper-header">
<h1 class="paper-title">{title}</h1>
<div class="author-block">
<span class="author-name">Aman Sachan</span><br/>
<span class="author-affiliation">{affiliation}</span><br/>
<span class="author-email"><code>{email}</code></span>
</div>
<div class="abstract-box">
<div class="abstract-title">Abstract</div>
<p class="abstract-text">{abstract_text}</p>
</div>
</div>"""
    
    final_markdown = html_header + "\n\n" + body_content.strip()
    
    # Premium LaTeX/NeurIPS Academic CSS stylesheet
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Inter:wght@400;500;600;700&family=Fira+Code:wght@400;500&display=swap');
    
    body {
        font-family: 'Lora', 'Georgia', serif;
        color: #000000;
        line-height: 1.6;
        font-size: 10.5pt;
        margin: 25mm 20mm 20mm 20mm;
    }
    
    .paper-header {
        text-align: center;
        margin-bottom: 30px;
    }
    
    .paper-title {
        font-family: 'Inter', sans-serif;
        font-size: 22pt;
        font-weight: 700;
        color: #000000;
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
        font-family: 'Inter', sans-serif;
        font-size: 12.5pt;
        font-weight: 600;
        color: #000000;
    }
    
    .author-affiliation {
        font-size: 10pt;
        color: #334155;
        font-style: italic;
    }
    
    .author-email {
        font-size: 9pt;
        color: #475569;
    }
    
    .abstract-box {
        max-width: 85%;
        margin: 25px auto 30px auto;
        padding-top: 15px;
        padding-bottom: 15px;
        border-top: 1.5px solid #000000;
        border-bottom: 1.5px solid #000000;
        text-align: justify;
    }
    
    .abstract-title {
        font-family: 'Inter', sans-serif;
        font-size: 10pt;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 8px;
        color: #000000;
        text-align: center;
    }
    
    .abstract-text {
        font-size: 9.5pt;
        line-height: 1.5;
        color: #000000;
        margin: 0;
    }
    
    h2, h3, h4 {
        font-family: 'Inter', sans-serif;
        color: #000000;
        font-weight: 700;
        margin-top: 2em;
        margin-bottom: 0.6em;
    }
    
    h2 {
        font-size: 13.5pt;
        border-bottom: 1.5px solid #000000;
        padding-bottom: 4px;
        margin-top: 2.2em;
    }
    
    h3 {
        font-size: 11.5pt;
        border-bottom: 1px solid #cbd5e1;
        padding-bottom: 2px;
    }
    
    p {
        margin-top: 0;
        margin-bottom: 1.2em;
        text-align: justify;
        text-justify: inter-word;
    }
    
    .equation {
        text-align: center;
        font-style: italic;
        margin: 20px 0;
        font-size: 11pt;
        font-family: 'Lora', 'Georgia', serif;
        color: #000000;
    }
    
    code {
        font-family: 'Fira Code', monospace;
        font-size: 8.5pt;
        background-color: #f1f5f9;
        color: #000000;
        padding: 1.5px 3.5px;
        border-radius: 3px;
    }
    
    pre {
        background-color: #f8fafc;
        color: #000000;
        border: 1.5px solid #cbd5e1;
        padding: 12px;
        border-radius: 4px;
        overflow-x: auto;
        margin-top: 1em;
        margin-bottom: 1.5em;
    }
    
    pre code {
        background-color: transparent;
        color: inherit;
        padding: 0;
        border-radius: 0;
        font-size: 8pt;
    }
    
    blockquote {
        margin: 1.5em 0;
        padding: 10px 20px;
        background-color: #f8fafc;
        border-left: 4px solid #000000;
        color: #111111;
        font-style: italic;
        border-radius: 0 4px 4px 0;
    }
    
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 2em 0;
        font-size: 9.5pt;
        border-top: 2px solid #000000;
        border-bottom: 2px solid #000000;
    }
    
    th, td {
        border: none;
        padding: 8px 12px;
        text-align: left;
    }
    
    th {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        color: #000000;
        border-bottom: 1.5px solid #000000;
        background-color: transparent;
    }
    
    td {
        border-bottom: 1px solid #cbd5e1;
    }
    
    tr:last-child td {
        border-bottom: none;
    }
    
    ul, ol {
        margin-top: 0;
        margin-bottom: 1.2em;
        padding-left: 20px;
    }
    
    li {
        margin-bottom: 0.5em;
    }
    
    hr {
        border: 0;
        border-top: 1.5px solid #cbd5e1;
        margin: 2.5em 0;
    }
    """
    
    pdf = MarkdownPdf(toc_level=0)
    pdf.add_section(Section(final_markdown), user_css=custom_css)
    
    print(f"Saving compiled paper PDF to {pdf_path}...")
    pdf.save(str(pdf_path))
    print("Success! CMF Research Paper PDF successfully compiled.")

if __name__ == "__main__":
    compile_paper()
