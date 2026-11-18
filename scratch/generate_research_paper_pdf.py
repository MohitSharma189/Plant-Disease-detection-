import os
import sys
import re
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

# ----------------------------------------------------------------------
# NumberedCanvas for Running Headers and Footers
# ----------------------------------------------------------------------
class AcademicCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(AcademicCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(AcademicCanvas, self).showPage()
        super(AcademicCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        page_num = self._pageNumber
        w, h = A4

        # Colors
        navy = colors.HexColor('#1B365D')
        slate = colors.HexColor('#555555')
        rule_color = colors.HexColor('#CCCCCC')

        if page_num > 1:
            # Header
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(navy)
            self.drawString(54, h - 36, "Plant Disease Screening System with MobileNetV3-Small")
            self.setFont("Helvetica-Oblique", 7.5)
            self.setFillColor(slate)
            self.drawRightString(w - 54, h - 36, "Research Manuscript — Comprehensive Revision")
            
            # Header rule
            self.setStrokeColor(rule_color)
            self.setLineWidth(0.5)
            self.line(54, h - 42, w - 54, h - 42)

        # Footer
        self.setStrokeColor(rule_color)
        self.setLineWidth(0.5)
        self.line(54, 45, w - 54, 45)

        self.setFont("Helvetica", 7.5)
        self.setFillColor(slate)
        self.drawString(54, 32, "Springer Nature Template Format — Implementation, Exploratory Results & Validation Gaps")
        self.drawRightString(w - 54, 32, f"Page {page_num} of {page_count}")

        self.restoreState()

print("AcademicCanvas defined successfully.")

