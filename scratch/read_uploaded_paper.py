import pypdf, textwrap

path = r"C:\Users\Mohit Sharma\.gemini\antigravity\brain\ef968360-f136-470a-a159-8b64d2823609\.user_uploaded\media_1794547302002_5bbab481.pdf"
reader = pypdf.PdfReader(path)
print(f"Total pages: {len(reader.pages)}\n{'='*60}")
for i, page in enumerate(reader.pages):
    text = page.extract_text() or ""
    print(f"\n{'='*60}\n=== PAGE {i+1} ===\n{'='*60}")
    print(text)

