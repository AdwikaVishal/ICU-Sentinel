import json
from pathlib import Path
from sklearn.model_selection import train_test_split

src = Path('data/splits/split_manifest.json')
out = Path('data/splits/split_manifest_fast_6000.json')
m = json.load(open(src))
train = [p for p in m['patients'] if p['split'] == 'train']
other = [p for p in m['patients'] if p['split'] != 'train']
labels = [bool(p['has_sepsis']) for p in train]
sel, _ = train_test_split(train, train_size=6000, random_state=42, stratify=labels)
selected = sel + other
selected.sort(key=lambda p: p['patient_id'])
m2 = dict(m)
m2['patients'] = selected
m2['n_patients_total'] = len(selected)
m2['n_patients_train'] = len(sel)
m2['n_records_train'] = sum(p['n_records'] for p in sel)
m2['n_sepsis_patients_train'] = sum(1 for p in sel if p['has_sepsis'])
m2['n_sepsis_positive_hours_train'] = sum(p['n_sepsis_positive_hours'] for p in sel)
m2['fast_protocol'] = {
    'enabled': True,
    'random_state': 42,
    'stratified_by': 'has_sepsis',
    'train_patients_requested': 6000,
    'train_patients_used': len(sel),
    'validation_and_test': 'full',
    'note': 'canonical reconstruction; exact P2 runtime manifest was overwritten',
}
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, 'w') as f:
    json.dump(m2, f, indent=2)
print('WROTE', out)
train_ids = set(p['patient_id'] for p in sel)
val_ids = set(p['patient_id'] for p in other if p['split'] == 'validation')
test_ids = set(p['patient_id'] for p in other if p['split'] == 'test')
print('train patients:', len(sel), 'records:', sum(p['n_records'] for p in sel))
print('val patients:', len(val_ids), 'test patients:', len(test_ids))
print('train/val overlap:', len(train_ids & val_ids))
print('train/test overlap:', len(train_ids & test_ids))
print('val/test overlap:', len(val_ids & test_ids))
