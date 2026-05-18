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
        color: #1e293b;
        line-height: 1.5;
        font-size: 10.5pt;
        margin: 20mm 20mm 20mm 20mm;
    }
    
    .paper-header {
        text-align: center;
        margin-bottom: 25px;
    }
    
    .paper-title {
        font-family: 'Inter', sans-serif;
        font-size: 21pt;
        font-weight: 700;
        color: #0f172a;
        margin-top: 15px;
        margin-bottom: 12px;
        line-height: 1.25;
        text-align: center;
    }
    
    .author-block {
        margin-bottom: 20px;
        line-height: 1.4;
        text-align: center;
    }
    
    .author-name {
        font-family: 'Inter', sans-serif;
        font-size: 12pt;
        font-weight: 600;
        color: #1e3a8a;
    }
    
    .author-affiliation {
        font-size: 9.5pt;
        color: #475569;
        font-style: italic;
    }
    
    .author-email {
        font-size: 8.5pt;
        color: #64748b;
    }
    
    .abstract-box {
        max-width: 90%;
        margin: 20px auto 25px auto;
        padding-top: 12px;
        padding-bottom: 12px;
        border-top: 1px solid #cbd5e1;
        border-bottom: 1px solid #cbd5e1;
        text-align: justify;
    }
    
    .abstract-title {
        font-family: 'Inter', sans-serif;
        font-size: 10pt;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
        color: #0f172a;
        text-align: center;
    }
    
    .abstract-text {
        font-size: 9.5pt;
        line-height: 1.45;
        color: #334155;
        margin: 0;
    }
    
    h2, h3, h4 {
        font-family: 'Inter', sans-serif;
        color: #0f172a;
        font-weight: 600;
        margin-top: 1.8em;
        margin-bottom: 0.5em;
    }
    
    h2 {
        font-size: 13pt;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 4px;
        color: #1e3a8a;
    }
    
    h3 {
        font-size: 11pt;
        color: #2563eb;
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
        font-size: 11pt;
        font-family: 'Lora', 'Georgia', serif;
        color: #0f172a;
    }
    
    code {
        font-family: 'Fira Code', monospace;
        font-size: 8.5pt;
        background-color: #f1f5f9;
        color: #0f172a;
        padding: 1.5px 3px;
        border-radius: 3px;
    }
    
    pre {
        background-color: #0f172a;
        color: #f8fafc;
        padding: 10px;
        border-radius: 5px;
        overflow-x: auto;
        margin-bottom: 1.2em;
    }
    
    pre code {
        background-color: transparent;
        color: inherit;
        padding: 0;
        border-radius: 0;
        font-size: 8pt;
    }
    
    blockquote {
        margin: 1.2em 0;
        padding: 8px 15px;
        background-color: #f8fafc;
        border-left: 3px solid #3b82f6;
        color: #334155;
        font-style: italic;
        border-radius: 0 4px 4px 0;
    }
    
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 1.5em 0;
        font-size: 9pt;
    }
    
    th, td {
        border-top: 1px solid #cbd5e1;
        border-bottom: 1px solid #cbd5e1;
        padding: 6px 10px;
        text-align: left;
    }
    
    th {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: #0f172a;
        background-color: #f8fafc;
    }
    
    ul, ol {
        margin-top: 0;
        margin-bottom: 1em;
        padding-left: 18px;
    }
    
    li {
        margin-bottom: 0.4em;
    }
    
    hr {
        border: 0;
        border-top: 1px solid #cbd5e1;
        margin: 2em 0;
    }
    """
    
    pdf = MarkdownPdf(toc_level=0)
    pdf.add_section(Section(final_markdown), user_css=custom_css)
    
    print(f"Saving compiled paper PDF to {pdf_path}...")
    pdf.save(str(pdf_path))
    print("Success! CMF Research Paper PDF successfully compiled.")

if __name__ == "__main__":
    compile_paper()
