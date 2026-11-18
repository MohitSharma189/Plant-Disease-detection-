"""
Executes the final revision PDF compilation and checks the page count.
"""

import os
import sys
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate
import pypdf

from build_final_revision_paper import AcademicNumberedCanvas, get_academic_styles
from paper_story_builder import build_full_story

def run_build():
    output_dir = "reports/research_paper/final_revision"
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, "plant_disease_research_paper.pdf")
    
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = get_academic_styles()
    printable_w = 487.27
    
    print("Building story elements...")
    story = build_full_story(styles, printable_w)
    
    print(f"Compiling PDF with {len(story)} flowable elements...")
    doc.build(story, canvasmaker=AcademicNumberedCanvas)
    
    if os.path.exists(pdf_path):
        reader = pypdf.PdfReader(pdf_path)
        page_count = len(reader.pages)
        size_kb = os.path.getsize(pdf_path) / 1024
        print(f"SUCCESS: Generated {pdf_path}")
        print(f"Page Count: {page_count} pages (Target: 18-22 pages)")
        print(f"File Size: {size_kb:.1f} KB")
        return page_count
    else:
        print("ERROR: PDF file was not created.")
        return 0

if __name__ == '__main__':
    run_build()

