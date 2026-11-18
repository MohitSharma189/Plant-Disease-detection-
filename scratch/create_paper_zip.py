import os
import zipfile

paper_dir = os.path.abspath("reports/research_paper")
zip_path = os.path.join(paper_dir, "plant_disease_research_paper.zip")

# Remove existing zip if present to avoid nesting
if os.path.exists(zip_path):
    os.remove(zip_path)

print(f"Creating ZIP archive at {zip_path}...")
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(paper_dir):
        for file in files:
            full_path = os.path.join(root, file)
            # Do not zip the zip itself
            if full_path == zip_path:
                continue
            rel_path = os.path.relpath(full_path, paper_dir)
            zipf.write(full_path, rel_path)
            print(f"  Added: {rel_path} ({os.path.getsize(full_path)} bytes)")

print(f"Successfully created ZIP. Total size: {os.path.getsize(zip_path)} bytes.")

