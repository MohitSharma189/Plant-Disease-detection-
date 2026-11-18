import re

def parse_bib(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    entries = []
    # Split by @
    chunks = content.split('@')
    for chunk in chunks[1:]:
        chunk = chunk.strip()
        if not chunk:
            continue
        first_brace = chunk.find('{')
        if first_brace == -1:
            continue
        entry_type = chunk[:first_brace].strip()
        body = chunk[first_brace+1:]
        # Find cite key
        comma = body.find(',')
        if comma == -1:
            continue
        cite_key = body[:comma].strip()
        body = body[comma+1:]
        
        fields = {}
        # Parse fields
        pattern = re.compile(r'(\w+)\s*=\s*[\{"](.*?)[\}"]\s*[,}]', re.DOTALL)
        for match in pattern.finditer(body):
            fields[match.group(1).lower()] = match.group(2).strip().replace('\n', ' ')
        entries.append((cite_key, entry_type, fields))
    return entries

parsed = parse_bib('reports/research_paper/references.bib')
print(f'Successfully parsed {len(parsed)} entries.')
for k, t, f in parsed[:5]:
    print(k, '-->', f.get('author', '')[:30], f.get('year', ''), f.get('title', '')[:30])

