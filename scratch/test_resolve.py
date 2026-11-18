import json, re, urllib.parse, time
from pathlib import Path

with open('reports/generated/combined_split_manifest.json', encoding='utf-8') as f:
    d = json.load(f)

def resolve_image_path(base_dir: Path, rel_path: str) -> Path:
    p = base_dir / rel_path
    if p.is_file():
        return p
    parts = Path(rel_path.replace('\\', '/')).parts
    cur = base_dir
    for part in parts:
        exact = cur / part
        if exact.exists():
            cur = exact
            continue
        sanitized = re.sub(r'[<>:"/\\|?*%]', '_', part).rstrip(' .')
        cand = cur / sanitized
        if cand.exists():
            cur = cand
            continue
        unquoted = urllib.parse.unquote(part)
        unquoted_cand = cur / unquoted
        if unquoted_cand.exists():
            cur = unquoted_cand
            continue
        unquoted_san = re.sub(r'[<>:"/\\|?*%]', '_', unquoted).rstrip(' .')
        unquoted_san_cand = cur / unquoted_san
        if unquoted_san_cand.exists():
            cur = unquoted_san_cand
            continue
        break
    if cur.is_file():
        return cur
    return p

data_dir = Path('data/processed/PlantVillage')
t0 = time.time()
pd_records = [r for r in d['records'] if r.get('source') == 'PlantDoc']
print(f'PlantDoc records: {len(pd_records)}')

missing = 0
for r in pd_records:
    base = (data_dir / Path(r.get('dataset_root', ''))).resolve()
    resolved = resolve_image_path(base, r['path'])
    if not resolved.is_file():
        missing += 1
        print('MISSING:', r['path'])

print(f'Done in {time.time()-t0:.2f}s, Missing PlantDoc: {missing}/{len(pd_records)}')
