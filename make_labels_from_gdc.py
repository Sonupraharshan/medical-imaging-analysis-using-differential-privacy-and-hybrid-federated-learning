#!/usr/bin/env python3
import pandas as pd
import os

# paths (edit if necessary)
metadata_csv = "data/metadata.csv"   # file_name, cases.submitter_id OR cases.0.submitter_id
clinical_csv = "data/clinical.csv"   # case_id / submitter_id, primary_diagnosis
out_csv = "data/labels.csv"

# load
md = pd.read_csv(metadata_csv, dtype=str)
clin = pd.read_csv(clinical_csv, dtype=str)

# heuristics to find the right columns
file_col = None
case_col_md = None
for c in md.columns:
    if "file" in c.lower() and "name" in c.lower():
        file_col = c
    if "submitter_id" in c.lower() or (c.lower().startswith("cases") and "submitter" in c.lower()):
        case_col_md = c
if file_col is None:
    raise SystemExit("Couldn't find file_name column in metadata CSV. Inspect columns: " + ", ".join(md.columns))
if case_col_md is None:
    if "case_id" in md.columns:
        case_col_md = "case_id"
    else:
        raise SystemExit("Couldn't find case submitter column in metadata CSV. Inspect columns: " + ", ".join(md.columns))

# find clinical case id column and label column
case_col_clin = None
label_col = None
for c in clin.columns:
    if "case" in c.lower() and ("submitter" in c.lower() or "id" in c.lower()):
        case_col_clin = c
    if "primary" in c.lower() and "diagnosis" in c.lower():
        label_col = c
    if "tumor" in c.lower() or "disease" in c.lower():
        label_col = label_col or c
if case_col_clin is None or label_col is None:
    raise SystemExit("Couldn't find case or label columns in clinical CSV. Inspect columns: " + ", ".join(clin.columns))

# build mapping case_id -> label code (LUAD->0, LUSC->1)
clin_map = {}
for _, row in clin.iterrows():
    cid = row[case_col_clin]
    label_text = str(row[label_col]).strip().upper()
    if "LUAD" in label_text or "ADENOCARCINOMA" in label_text:
        code = 0
    elif "LUSC" in label_text or "SQUAMOUS" in label_text:
        code = 1
    else:
        code = -1
    clin_map[cid] = code

# join metadata to clinical
rows = []
for _, r in md.iterrows():
    fname = r[file_col]
    caseid = r[case_col_md]
    slide_id = os.path.splitext(fname)[0]
    lab = clin_map.get(caseid, -1)
    rows.append({"slide_id": slide_id, "label": lab})

out = pd.DataFrame(rows)
out.to_csv(out_csv, index=False)
print("Wrote", out_csv, "— inspect and remove label=-1 entries if necessary.")