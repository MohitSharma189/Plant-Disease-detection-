import sys
libs = ['fpdf', 'fpdf2', 'reportlab', 'weasyprint', 'pdfkit']
for lib in libs:
    try:
        m = __import__(lib)
        print(f"OK: {lib} {getattr(m, '__version__', 'unknown')}")
    except ImportError:
        print(f"MISSING: {lib}")

