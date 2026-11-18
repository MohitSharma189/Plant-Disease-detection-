"""
Full PDF Builder for the Final Revision Manuscript.
Targets: reports/research_paper/final_revision/plant_disease_research_paper.pdf
Target length: 18-20 pages.
Ensures perfect styling, strict academic tone, zero synthetic data, verified values.
"""

import os
import sys
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage,
    KeepTogether, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas

import final_paper_data as fpd
import final_paper_text as fpt

class AcademicNumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 7.5)
        self.setFillColor(colors.HexColor('#475569'))
        
        # Running Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 802, "Springer Nature Applied Intelligence  |  Plant Disease Screening System with MobileNetV3-Small")
            self.drawRightString(541, 802, "Sharma et al. (2026)")
            self.setStrokeColor(colors.HexColor('#CBD5E1'))
            self.setLineWidth(0.5)
            self.line(54, 796, 541, 796)
        
        # Running Footer (all pages)
        self.setStrokeColor(colors.HexColor('#E2E8F0'))
        self.setLineWidth(0.5)
        self.line(54, 46, 541, 46)
        self.drawString(54, 34, "Research Article  |  Cryptographically Verified Provenance Pipeline  |  Phase 6 Final Revision")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(541, 34, page_str)
        self.restoreState()

def get_academic_styles():
    ss = getSampleStyleSheet()
    styles = {}
    
    styles['JournalBanner'] = ParagraphStyle(
        'JournalBanner',
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor('#1B365D')
    )
    styles['JournalMeta'] = ParagraphStyle(
        'JournalMeta',
        fontName='Helvetica',
        fontSize=6.5,
        leading=8.5,
        textColor=colors.HexColor('#64748B')
    )
    styles['PaperTitle'] = ParagraphStyle(
        'PaperTitle',
        fontName='Helvetica-Bold',
        fontSize=14.5,
        leading=18,
        textColor=colors.HexColor('#0F172A'),
        spaceAfter=8
    )
    styles['AuthorName'] = ParagraphStyle(
        'AuthorName',
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=2
    )
    styles['Affiliation'] = ParagraphStyle(
        'Affiliation',
        fontName='Helvetica',
        fontSize=7,
        leading=9.5,
        textColor=colors.HexColor('#475569'),
        spaceAfter=4
    )
    styles['AuthorNotes'] = ParagraphStyle(
        'AuthorNotes',
        fontName='Helvetica-Oblique',
        fontSize=6.5,
        leading=8.5,
        textColor=colors.HexColor('#64748B'),
        spaceAfter=8
    )
    styles['AbstractHeading'] = ParagraphStyle(
        'AbstractHeading',
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=10.5,
        textColor=colors.HexColor('#0F172A'),
        spaceAfter=3
    )
    styles['AbstractBody'] = ParagraphStyle(
        'AbstractBody',
        fontName='Helvetica',
        fontSize=7.5,
        leading=10.5,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor('#1E293B')
    )
    styles['Keywords'] = ParagraphStyle(
        'Keywords',
        fontName='Helvetica',
        fontSize=7,
        leading=9.5,
        textColor=colors.HexColor('#334155'),
        spaceBefore=4
    )
    styles['Sec1'] = ParagraphStyle(
        'Sec1',
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=13,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=12,
        spaceAfter=4,
        keepWithNext=True
    )
    styles['Sec2'] = ParagraphStyle(
        'Sec2',
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#1E293B'),
        spaceBefore=8,
        spaceAfter=3,
        keepWithNext=True
    )
    styles['Sec3'] = ParagraphStyle(
        'Sec3',
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor('#334155'),
        spaceBefore=5,
        spaceAfter=2,
        keepWithNext=True
    )
    styles['Body'] = ParagraphStyle(
        'Body',
        fontName='Helvetica',
        fontSize=7.5,
        leading=10.8,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=6
    )
    styles['TableCaption'] = ParagraphStyle(
        'TableCaption',
        fontName='Helvetica-Bold',
        fontSize=7,
        leading=9,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=6.5,
        spaceAfter=2.5,
        keepWithNext=True
    )
    styles['FigureCaption'] = ParagraphStyle(
        'FigureCaption',
        fontName='Helvetica',
        fontSize=6.5,
        leading=8.5,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor('#334155'),
        spaceBefore=3,
        spaceAfter=6
    )
    styles['TableHeader'] = ParagraphStyle(
        'TableHeader',
        fontName='Helvetica-Bold',
        fontSize=6.5,
        leading=8,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#FFFFFF')
    )
    styles['TableCell'] = ParagraphStyle(
        'TableCell',
        fontName='Helvetica',
        fontSize=6.2,
        leading=7.8,
        textColor=colors.HexColor('#1E293B')
    )
    styles['TableCellBold'] = ParagraphStyle(
        'TableCellBold',
        fontName='Helvetica-Bold',
        fontSize=6.2,
        leading=7.8,
        textColor=colors.HexColor('#0F172A')
    )
    styles['FormulaBox'] = ParagraphStyle(
        'FormulaBox',
        fontName='Helvetica-Oblique',
        fontSize=7,
        leading=9.5,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#0F172A')
    )
    styles['Callout'] = ParagraphStyle(
        'Callout',
        fontName='Helvetica',
        fontSize=7,
        leading=9.5,
        textColor=colors.HexColor('#1E293B')
    )
    styles['RefEntry'] = ParagraphStyle(
        'RefEntry',
        fontName='Helvetica',
        fontSize=6.5,
        leading=8.5,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor('#1E293B'),
        leftIndent=12,
        firstLineIndent=-12,
        spaceAfter=3.2
    )
    return styles

def make_callout(text, styles, width, bg_color='#F1F5F9', border_color='#94A3B8'):
    p = Paragraph(text, styles['Callout'])
    t = Table([[p]], colWidths=[width])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor(bg_color)),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor(border_color)),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    return t

def make_formula_table(equation_text, eq_num, styles, width):
    eq_p = Paragraph(equation_text, styles['FormulaBox'])
    num_p = Paragraph(f"({eq_num})", ParagraphStyle('EqNum', fontName='Helvetica', fontSize=7, leading=9, alignment=TA_RIGHT))
    t = Table([[eq_p, num_p]], colWidths=[width - 35, 35])
    t.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 2),
    ]))
    return t
