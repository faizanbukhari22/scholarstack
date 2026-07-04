import os
import markdown
from fpdf import FPDF

# fpdf2's core fonts (helvetica) only support latin-1. LLM output routinely
# contains curly quotes, dashes, arrows, and math symbols, which would raise
# FPDFUnicodeEncodingException mid-render. Map the common cases to ASCII and
# replace anything else so PDF export never crashes on real lectures.
_ASCII_MAP = {
    "‘": "'", "’": "'", "‚": "'",
    "“": '"', "”": '"', "„": '"',
    "–": "-", "—": "-", "−": "-",
    "…": "...", "•": "-", "·": "-",
    "→": "->", "←": "<-", "↔": "<->",
    "×": "x", "÷": "/", "±": "+/-",
    " ": " ", "​": "", "﻿": "",
}


def _sanitize_for_latin1(text: str) -> str:
    for src, dst in _ASCII_MAP.items():
        text = text.replace(src, dst)
    return text.encode("latin-1", "replace").decode("latin-1")

class StudyGuidePDF(FPDF):
    def header(self):
        # Set top margin border
        self.set_font("helvetica", "B", 8)
        self.set_text_color(100, 116, 139)  # Slate 500
        self.cell(0, 10, "EduAgent-OS -- Autonomous Lecture Notes", align="L")
        self.cell(0, 10, "Study Guide", align="R", new_x="LMARGIN", new_y="NEXT")
        
        # Draw top thin rule
        self.set_draw_color(226, 232, 240)  # Slate 200
        self.line(self.l_margin, 18, self.w - self.r_margin, 18)
        self.ln(5)

    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(100, 116, 139)  # Slate 500
        # Print page number and total pages
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

def compile_markdown_to_pdf(md_path: str, pdf_path: str):
    """Converts a Markdown notes file into a clean, formatted PDF document.
    
    This function reads a Markdown file, converts it into basic HTML, and 
    uses fpdf2 to write the parsed HTML structure into a PDF document.
    """
    if not os.path.exists(md_path):
        raise FileNotFoundError(f"Markdown file not found at: {md_path}")

    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    # Convert Markdown to HTML with tables support
    html_content = markdown.markdown(_sanitize_for_latin1(md_content), extensions=['tables'])

    # Create PDF document
    pdf = StudyGuidePDF(orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    
    # Configure auto page breaks
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Set standard font
    pdf.set_font("helvetica", size=10)
    
    # Write parsed HTML elements to PDF
    pdf.write_html(html_content)
    
    # Output to target path
    pdf.output(pdf_path)
    print(f"[PDF Generator] Successfully compiled PDF at: {pdf_path}")
