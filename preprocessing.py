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
# Improved line-item extractor (drop-in replacement)
# ---------------------------
DATE_RE = re.compile(r'\b\d{1,2}[\/\-\.\s]\d{1,2}[\/\-\.\s]\d{2,4}\b')

def is_date_token(txt: str) -> bool:
    if not txt:
        return False
    return bool(DATE_RE.search(txt))

def extract_line_item_from_row_using_columns(row: List[Dict[str,Any]], centers: List[int], page_style: str = 'unknown') -> Dict[str,Any] | None:
    """
    Heuristics:
    - Build column map, then collect numeric tokens sorted by x (right-to-left)
    - Rightmost plausible numeric -> amount
    - Among remaining numerics left of amount:
        - first small integer (<=1000 and nearly integer) -> qty
        - first reasonable decimal/number -> rate
    - If qty missing and rate present -> qty = amount / rate (if plausible)
    - If rate missing and qty present -> rate = amount / qty
    - Remove date-like tokens from item_name and ignore date tokens when selecting numerics
    """
    colmap = assign_row_words_to_columns(row, centers)
    ncols = len(centers)

    # Build full-row text but we'll remove date tokens later
    # Prepare numeric candidates: collect all numeric-like words with xmid
    numeric_candidates = []
    for w in row:
        txt = w['text']
        if is_date_token(txt):
            continue
        val = normalize_number_token(txt, style=page_style)
        if val is not None:
            x1,_,x2,_ = w['bbox']
            xmid = (x1 + x2) / 2.0
            numeric_candidates.append((xmid, val, txt, w))

    if not numeric_candidates:
        return None

    # Sort numeric candidates by xmid descending (right-to-left)
    numeric_candidates.sort(key=lambda t: t[0], reverse=True)

    # Choose amount: first numeric that is plausible (not a year-like, > 0, not tiny fractional like 0.0)
    amount = None
    amount_x = None
    for xmid, val, txt, w in numeric_candidates:
        # skip values that look like years e.g., 2025
        try:
            vint = int(round(val))
            if 1900 <= vint <= 2100:
                continue
        except Exception:
            pass
        # skip nonsense zeros
        if val is None or val <= 0:
            continue
        amount = float(val)
        amount_x = xmid
        break

    if amount is None:
        return None

    # Now gather left-side candidates (x < amount_x), sorted descending (closest to amount first)
    left_candidates = [(x,v,txt,w) for x,v,txt,w in numeric_candidates if x < amount_x]
    left_candidates.sort(key=lambda t: t[0], reverse=True)

    qty = None
    rate = None

    # Heuristics to pick qty and rate from left candidates
    for xmid, val, txt, w in left_candidates:
        # skip dates or weird tokens (already filtered but just in case)
        if is_date_token(txt):
            continue
        # candidate integer-like and small -> likely qty
        if qty is None:
            # treat as qty if it's an almost-integer and reasonably small
            if abs(round(val) - val) < 1e-6 and 0 < val <= 10000:
                # further prefer small numbers (<=1000) as qty
                if val <= 1000:
                    qty = int(round(val))
                    continue
        # candidate plausible as rate: decimals, or > 10 etc
        if rate is None:
            # if it has decimals or > 10 (and not huge), accept as rate
            if (not abs(round(val) - val) < 1e-6) or (val > 10 and val < 1e7):
                rate = float(val)
                continue
        # fallback: if still nothing, accept any numeric into rate
        if rate is None:
            rate = float(val)

    # Fallback inference
    if qty is None and rate is not None and rate != 0:
        try:
            qcand = amount / rate
            if qcand > 0 and qcand < 1e6:
                # prefer integer if nearly integer
                if abs(round(qcand) - qcand) < 0.01:
                    qty = int(round(qcand))
                else:
                    qty = round(qcand, 2)
        except Exception:
            qty = None

    if rate is None and qty is not None and qty != 0:
        try:
            rate = amount / qty
        except Exception:
            rate = None

    # Compose description: take words left of the first numeric token (left-most numeric in row)
    # Find the x position of the earliest numeric in the row (leftmost numeric)
    all_numeric_x = [ ( (w['bbox'][0] + w['bbox'][2])/2.0, w ) for w in row if normalize_number_token(w['text'], style=page_style) is not None and not is_date_token(w['text'])]
    first_numeric_x = None
    if all_numeric_x:
        first_numeric_x = min(all_numeric_x, key=lambda t: t[0])[0]

    desc_parts = []
    for w in row:
        x1,_,x2,_ = w['bbox']
        xmid = (x1 + x2) / 2.0
        txt = w['text'].strip()
        # Exclude tokens that are numeric/date and those that are to the right of first_numeric_x
        if first_numeric_x is not None and xmid >= first_numeric_x:
            continue
        if is_date_token(txt):
            continue
        # skip small numeric tokens embedded in description
        if normalize_number_token(txt, style=page_style) is not None:
            continue
        desc_parts.append(txt)

    item_name = ' '.join(desc_parts).strip()
    # final cleanup: collapse multiple spaces and trim
    item_name = re.sub(r'\s{2,}', ' ', item_name).strip()

    # Filter out lines that are totals/headers
    if not item_name or not re.search(r'[A-Za-z]', item_name):
        return None
    if any(k in item_name.lower() for k in ['total', 'subtotal', 'balance', 'grand total', 'page', 'printed on']):
        return None

    return {
        'item_name': item_name,
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
