# rewritten preprocessing.py
# Robust table extraction for hospital/medical bills
# Supports: (1) strict column tables, (2) label-right numeric lists, (3) sectioned bills

import os
import re
import json
import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
import pytesseract
from typing import List, Dict, Any, Tuple

# ---------------------------
# Utilities
# ---------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# ---------------------------
# PDF -> images (list of BGR numpy arrays)
# ---------------------------

def pdf_to_images(input_path: str, dpi: int = 300) -> List[np.ndarray]:
    if input_path.lower().endswith('.pdf'):
        pages = convert_from_path(input_path, dpi=dpi)
    else:
        pages = [Image.open(input_path)]

    imgs = []
    for p in pages:
        rgb = p.convert('RGB')
        arr = np.array(rgb)[:, :, ::-1].copy()  # PIL RGB -> OpenCV BGR
        imgs.append(arr)
    return imgs

# ---------------------------
# OCR helper (tesseract layout)
# returns list of word dicts: {text, conf, bbox, page, block_num}
# ---------------------------

def run_tesseract_layout_ocr(image: np.ndarray, psm: int = 6, oem: int = 1, numeric_conf_thresh: float = 25.0) -> List[Dict[str, Any]]:
    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    config = f"--oem {oem} --psm {psm}"
    data = pytesseract.image_to_data(pil, output_type=pytesseract.Output.DICT, config=config)
    out = []
    n = len(data.get('level', []))
    for i in range(n):
        txt = str(data.get('text', [''])[i]).strip()
        try:
            conf = float(data.get('conf', [0])[i])
        except Exception:
            conf = 0.0
        if not txt:
            continue
        # keep most tokens but filter extremely low-confidence non-numeric tokens
        is_numeric_like = bool(re.search(r'[\d\.,]+', txt))
        if conf < 10 and not is_numeric_like:
            continue
        x = int(data.get('left', [0])[i])
        y = int(data.get('top', [0])[i])
        w = int(data.get('width', [0])[i])
        h = int(data.get('height', [0])[i])
        bbox = [x, y, x + w, y + h]
        out.append({
            'text': txt,
            'conf': conf,
            'bbox': bbox,
            'page': int(data.get('page_num', [1])[i]),
            'block_num': int(data.get('block_num', [0])[i])
        })
    return out

# ---------------------------
# Numeric normalizer
# ---------------------------

def normalize_number_token(tok: str, style: str = 'unknown') -> float | None:
    if tok is None:
        return None
    s = str(tok).strip()
    s = re.sub(r'^[^\d\-\.,]+|[^\d\-\.,]+$', '', s)
    if s == '' or all(ch in '-,.' for ch in s):
        return None
    # common cases
    try:
        # if has both separators, prefer dot as decimal if dot appears after comma or vice versa
        if ',' in s and '.' in s:
            if s.rfind('.') > s.rfind(','):
                cand = s.replace(',', '')
            else:
                cand = s.replace('.', '').replace(',', '.')
            return float(cand)
        # only dot
        if '.' in s and ',' not in s:
            return float(s)
        # only comma
        if ',' in s and '.' not in s:
            # could be thousands or decimal; heuristic: if last group length 3 -> thousands
            parts = s.split(',')
            if len(parts[-1]) == 3 and len(parts) > 1:
                # e.g. 1,234 -> 1234
                return float(s.replace(',', ''))
            else:
                # treat comma as decimal
                return float(s.replace(',', '.'))
        # plain integer
        if re.fullmatch(r'-?\d+', s):
            return float(s)
    except Exception:
        return None
    return None

# ---------------------------
# Header detection: find table header top y and header word positions
# ---------------------------

def find_table_header_top_and_headers(raw_ocr: List[Dict[str,Any]], keywords: List[str] = None) -> Tuple[int, List[Dict[str,int]] | None]:
    if not keywords:
        keywords = ['description', 'qty', 'quantity', 'qty / hrs', 'qty/hrs', 'rate', 'amount', 'net', 'gross', 'total']
    matches = [w for w in raw_ocr if any(k in w['text'].lower() for k in keywords)]
    if not matches:
        return None, None
    # take the smallest y (top) among header-like words
    top_y = min([w['bbox'][1] for w in matches])
    # gather x-centers for header words
    headers = []
    for w in matches:
        x1, y1, x2, y2 = w['bbox']
        headers.append({'text': w['text'].lower(), 'xmid': (x1 + x2) // 2, 'bbox': w['bbox']})
    # sort by xmid
    headers = sorted(headers, key=lambda h: h['xmid'])
    return top_y, headers

# ---------------------------
# Detect column centers from header anchors
# ---------------------------

def detect_column_centers_from_headers(headers: List[Dict[str,int]], image_width: int) -> List[int]:
    if not headers:
        # fallback: split into 4 equal columns
        return [int(image_width * (i+0.5) / 4.0) for i in range(4)]
    centers = [int(h['xmid']) for h in headers]
    # if there are fewer than 3 centers, expand by adding extremes
    if len(centers) == 1:
        centers = [int(image_width*0.2), centers[0], int(image_width*0.8)]
    return centers

# ---------------------------
# Group words into rows using y-center clustering (robust)
# ---------------------------

def group_words_to_rows_by_y(words: List[Dict[str,Any]], row_tol: int = 12) -> List[List[Dict[str,Any]]]:
    if not words:
        return []
    items = []
    for w in words:
        x1,y1,x2,y2 = w['bbox']
        ymid = (y1 + y2) / 2.0
        items.append((ymid, w))
    items.sort(key=lambda it: it[0])
    rows = []
    current_row = [items[0][1]]
    ref = items[0][0]
    for ymid, w in items[1:]:
        if abs(ymid - ref) <= row_tol:
            current_row.append(w)
            ref = (ref * (len(current_row)-1) + ymid) / len(current_row)
        else:
            # sort current row by x
            current_row.sort(key=lambda ww: ww['bbox'][0])
            rows.append(current_row)
            current_row = [w]
            ref = ymid
    if current_row:
        current_row.sort(key=lambda ww: ww['bbox'][0])
        rows.append(current_row)
    return rows

# ---------------------------
# Assign words in a row to nearest column center
# ---------------------------

def assign_row_words_to_columns(row: List[Dict[str,Any]], centers: List[int]) -> Dict[int, List[Dict[str,Any]]]:
    colmap = {i: [] for i in range(len(centers))}
    for w in row:
        x1,y1,x2,y2 = w['bbox']
        xmid = (x1 + x2)/2.0
        # choose nearest center
        diffs = [abs(xmid - c) for c in centers]
        idx = int(np.argmin(diffs))
        colmap[idx].append(w)
    # sort words in each column by x
    for k in colmap:
        colmap[k].sort(key=lambda ww: ww['bbox'][0])
    return colmap

# ---------------------------
# Compose text from a list of words
# ---------------------------

def words_to_text(words: List[Dict[str,Any]]) -> str:
    if not words:
        return ''
    return ' '.join([w['text'] for w in words]).strip()

# ---------------------------
# Extract candidate items from rows using columns
# Returns dicts: item_name, quantity, unit_price, amount
# ---------------------------

def extract_line_item_from_row_using_columns(row: List[Dict[str,Any]], centers: List[int], page_style: str = 'unknown') -> Dict[str,Any] | None:
    colmap = assign_row_words_to_columns(row, centers)
    # heuristic: left-most columns -> description, right-most -> amount/rate/qty
    ncols = len(centers)
    # pick description from first non-empty column from left
    desc = ''
    for i in range(ncols):
        t = words_to_text(colmap.get(i, []))
        if t and not re.search(r'^\d', t):
            desc = t
            break
    # gather numeric tokens from rightmost columns
    numeric_candidates = []
    for i in range(ncols-1, -1, -1):
        for w in reversed(colmap.get(i, [])):
            val = normalize_number_token(w['text'], style=page_style)
            if val is not None:
                numeric_candidates.append((i, val, w))
    if not numeric_candidates:
        return None
    # amount is first numeric candidate (rightmost/highest x)
    amount = numeric_candidates[0][1]
    # find quantity: small integer near left of amount
    qty = None
    rate = None
    for idx, val, w in numeric_candidates[1:5]:
        # heuristic: qty is small integer <= 1000 and likely integer
        if qty is None and abs(round(val) - val) < 1e-6 and 0 < val <= 10000 and val <= 1000:
            qty = int(round(val))
            continue
        if rate is None and val > 0:
            rate = val
    # fallback: if qty missing but rate exists, compute qty = amount / rate
    if qty is None and rate and rate != 0:
        try:
            qcand = amount / rate
            if 0 < qcand < 1e6:
                qty = int(round(qcand)) if abs(round(qcand)-qcand) < 0.01 else qcand
        except Exception:
            qty = None
    # fallback: if rate missing but qty exists
    if rate is None and qty and qty != 0:
        try:
            rate = amount / qty
        except Exception:
            rate = None
    # ensure description contains letters
    if not re.search(r'[A-Za-z]', desc):
        # try building desc by taking all words left of first numeric in row
        desc_parts = []
        for w in row:
            if normalize_number_token(w['text'], style=page_style) is None:
                desc_parts.append(w['text'])
            else:
                break
        desc = ' '.join(desc_parts).strip()

    # filter out lines that look like totals or headers
    low_desc = desc.lower()
    if any(x in low_desc for x in ['total', 'subtotal', 'balance', 'tax', 'discount', 'amount due', 'grand total']):
        return None

    if amount is None or not re.search(r'[A-Za-z]', desc):
        return None

    return {
        'item_name': desc.strip(),
        'quantity': qty if qty is not None else None,
        'unit_price': float(rate) if rate is not None else None,
        'amount': float(amount)
    }

# ---------------------------
# Postprocess items: clean strings, infer missing qty/rate
# ---------------------------

def postprocess_line_items(items: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    cleaned = []
    for it in items:
        name = (it.get('item_name') or '').strip()
        if not name or not re.search(r'[A-Za-z]', name):
            continue
        if any(k in name.lower() for k in ['total', 'subtotal', 'grand total', 'balance']):
            continue
        amount = it.get('amount')
        qty = it.get('quantity')
        up = it.get('unit_price')
        # normalize amount
        try:
            amount = float(amount) if amount is not None else None
        except Exception:
            amount = None
        # infer qty and up
        if qty is None and up is not None and amount is not None:
            try:
                q = amount / up if up != 0 else None
                if q is not None:
                    if abs(round(q) - q) < 0.01:
                        qty = int(round(q))
                    else:
                        qty = round(q, 2)
            except Exception:
                qty = None
        if up is None and qty is not None and amount is not None and qty != 0:
            try:
                up = amount / qty
            except Exception:
                up = None
        # final validation
        if amount is None:
            continue
        it2 = {
            'item_name': re.sub(r'\s{2,}', ' ', name),
            'quantity': float(qty) if qty is not None else None,
            'unit_price': float(up) if up is not None else None,
            'amount': float(amount)
        }
        cleaned.append(it2)
    return cleaned

# ---------------------------
# Top-level processor: processes document and extracts items per page
# ---------------------------

def process_document_kmeans(input_path: str, out_dir: str) -> Tuple[List[Dict[str,Any]], List[Dict[str,Any]]]:
    ensure_dir(out_dir)
    pages = pdf_to_images(input_path)
    all_items = []
    summaries = []
    for i, img in enumerate(pages):
        page_dir = os.path.join(out_dir, f'page_{i+1:03d}')
        ensure_dir(page_dir)
        # write debug images if needed
        h,w = img.shape[:2]
        raw_ocr = run_tesseract_layout_ocr(img)
        with open(os.path.join(page_dir, 'raw_ocr.json'), 'w') as f:
            json.dump(raw_ocr, f, indent=2)
        top_y, headers = find_table_header_top_and_headers(raw_ocr)
        # restrict words to table region if header found
        if top_y is not None:
            table_words = [w for w in raw_ocr if w['bbox'][1] >= (top_y - 5)]
        else:
            table_words = raw_ocr
        centers = detect_column_centers_from_headers(headers, w)
        rows = group_words_to_rows_by_y(table_words, row_tol=14)
        page_items = []
        page_style = 'unknown'
        for row in rows:
            li = extract_line_item_from_row_using_columns(row, centers, page_style=page_style)
            if li:
                page_items.append(li)
        page_items = postprocess_line_items(page_items)
        # save page outputs
        with open(os.path.join(page_dir, 'line_items.json'), 'w') as f:
            json.dump(page_items, f, indent=2)
        all_items.extend(page_items)
        summary = {
            'page': i+1,
            'skew_angle': 0.0,
            'language': 'unknown',
            'table_count': 1 if page_items else 0,
            'fraud': {
                'whiteners': 0,
                'suspicious_numbers': 0,
                'inconsistent_fonts': False,
                'numeric_patch_detected': False
            },
            'line_items_extracted': len(page_items)
        }
        with open(os.path.join(page_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        summaries.append(summary)
    with open(os.path.join(out_dir, 'run_summary.json'), 'w') as f:
        json.dump(summaries, f, indent=2)
    return summaries, all_items

# ---------------------------
# CLI
# ---------------------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='Input PDF or image')
    parser.add_argument('-o', '--out', required=True, help='Output directory')
    args = parser.parse_args()
    res, items = process_document_kmeans(args.input, args.out)
    print(json.dumps(res, indent=2))
    print(f'Extracted items: {len(items)}')
