# preprocessing_table_kmeans.py
import cv2
import numpy as np
import os
import pytesseract
import argparse
import json
import imutils
import re
from PIL import Image
from pdf2image import convert_from_path
from tqdm import tqdm
from langdetect import detect, LangDetectException
from typing import List, Dict, Any, Tuple

# ---------------------------
# Utilities
# ---------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# ---------------------------
# PDF → Image
# ---------------------------
def pdf_to_images(pdf_path, dpi=300):
    if pdf_path.lower().endswith('.pdf'):
        pages = convert_from_path(pdf_path, dpi=dpi)
    else:
        pages = [Image.open(pdf_path)]

    imgs = []
    for p in pages:
        rgb = p.convert('RGB')
        arr = np.array(rgb)[:, :, ::-1].copy()
        imgs.append(arr)
    return imgs

# ---------------------------
# Skew, Denoise, Segmentation (unchanged)
# ---------------------------
def deskew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=200, maxLineGap=20)
    angles = []
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            angles.append(np.degrees(np.arctan2(y2-y1, x2-x1)))
    if not angles:
        return image, 0.0
    median_angle = np.median(angles)
    (h,w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w//2,h//2), median_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w,h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated, round(float(median_angle),2)

def remove_noise(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    denoised = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)

def segment_page(image):
    h,w = image.shape[:2]
    header = (0,0,w,int(0.12*h))
    body = (0,int(0.12*h),w,int(0.88*h))
    footer = (0,int(0.88*h),w,h)
    return {"header": header, "body": body, "footer": footer}

# ---------------------------
# OCR helper (keeps numeric tokens)
# ---------------------------
def run_tesseract_layout_ocr(denoised_image, numeric_conf_thresh=30):
    pil_img = Image.fromarray(cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB))
    config = "--oem 1 --psm 6"
    data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT, config=config)
    structured_output = []
    n_boxes = len(data.get('level', []))
    for i in range(n_boxes):
        raw_text = str(data.get('text', [''])[i]).strip()
        try:
            conf = float(data.get('conf', [0])[i])
        except Exception:
            conf = 0.0
        is_numeric_like = bool(re.search(r'[\d\.,]+', raw_text))
        keep = False
        if raw_text:
            if conf > 50:
                keep = True
            elif is_numeric_like and conf >= numeric_conf_thresh:
                keep = True
        if not keep:
            continue
        x = int(data.get('left', [0])[i])
        y = int(data.get('top', [0])[i])
        w = int(data.get('width', [0])[i])
        h = int(data.get('height', [0])[i])
        bbox = [x, y, x + w, y + h]
        structured_output.append({
            "text": raw_text,
            "confidence": conf,
            "bbox": bbox,
            "page": int(data.get('page_num', [1])[i]),
            "block_num": int(data.get('block_num', [1])[i])
        })
    return structured_output

# ---------------------------
# Decimal-style detection & numeric normalizer (auto-detect per page)
# ---------------------------
def detect_decimal_style(raw_ocr_data: List[Dict[str, Any]], debug: bool = False) -> str:
    tokens = [w.get("text", "").strip() for w in raw_ocr_data if isinstance(w.get("text", ""), str) and w.get("text", "").strip()]
    dot_decimal = 0
    comma_decimal = 0
    thousands_commas = 0
    thousands_dots = 0
    both_separators = 0

    p_dot_dec = re.compile(r'^\d{1,3}(?:,\d{3})*(?:\.\d{1,3})?$')   # 1,234.56 or 1234.56
    p_comma_dec = re.compile(r'^\d{1,3}(?:\.\d{3})*(?:,\d{1,3})?$') # 1.234,56 or 1234,56
    p_commas_thousands = re.compile(r'^\d{1,3}(?:,\d{3})+$')        # 1,234
    p_dots_thousands = re.compile(r'^\d{1,3}(?:\.\d{3})+$')        # 1.234

    MAX_TOKENS = 200
    for t in tokens[:MAX_TOKENS]:
        if ',' in t and '.' in t:
            both_separators += 1
            if p_dot_dec.match(t):
                dot_decimal += 1
            if p_comma_dec.match(t):
                comma_decimal += 1
            continue
        if p_dot_dec.match(t):
            dot_decimal += 1
        if p_comma_dec.match(t):
            comma_decimal += 1
        if p_commas_thousands.match(t):
            thousands_commas += 1
        if p_dots_thousands.match(t):
            thousands_dots += 1

    if debug:
        print("detect_decimal_style:", dot_decimal, comma_decimal, thousands_commas, thousands_dots, both_separators)

    if comma_decimal >= max(2, dot_decimal * 2):
        return 'comma'
    if dot_decimal >= max(2, comma_decimal * 2):
        return 'dot'
    if thousands_commas >= max(2, thousands_dots * 2) and dot_decimal >= 1:
        return 'dot'
    if thousands_dots >= max(2, thousands_commas * 2) and comma_decimal >= 1:
        return 'comma'
    if both_separators > 0:
        if dot_decimal > comma_decimal:
            return 'dot'
        if comma_decimal > dot_decimal:
            return 'comma'
    return 'unknown'

def detect_and_set_page_style(raw_ocr_data: List[Dict[str, Any]], debug: bool = False) -> str:
    style = detect_decimal_style(raw_ocr_data, debug=debug)
    return style

def normalize_number_token(tok: str, style: str = 'unknown'):
    if not tok or not isinstance(tok, str):
        return None
    s = tok.strip()
    s = re.sub(r'^[^\d\-\.,]+|[^\d\-\.,]+$', '', s)
    if s == '' or all(ch in '-,.' for ch in s):
        return None
    def try_float(x):
        try:
            return float(x)
        except:
            return None
    if ',' in s and '.' in s:
        cand = s.replace(',', '')
        return try_float(cand)
    if style == 'dot':
        cand = s.replace(',', '')
        return try_float(cand)
    if style == 'comma':
        if '.' in s:
            cand = s.replace('.', '').replace(',', '.')
        else:
            cand = s.replace(',', '.')
        return try_float(cand)
    # unknown style: conservative
    if '.' in s and ',' not in s:
        return try_float(s)
    if ',' in s and '.' not in s:
        parts = s.split(',')
        if len(parts) > 1 and all(len(p) == 3 for p in parts[1:]):
            return try_float(s.replace(',', ''))
        return None
    if re.fullmatch(r'-?\d+', s):
        return float(s)
    return None

# ---------------------------
# Table block detection (uses Tesseract block_num)
# ---------------------------
def find_table_block(raw_ocr_data: List[Dict[str, Any]], keywords=('item','description','price','qty','total')):
    blocks = {}
    for w in raw_ocr_data:
        b = w.get('block_num', 0)
        if b not in blocks:
            blocks[b] = {"words": [], "texts": [], "bbox": [1e9, 1e9, 0, 0]}
        blocks[b]["words"].append(w)
        blocks[b]["texts"].append(w['text'].lower())
        x1,y1,x2,y2 = w['bbox']
        bb = blocks[b]["bbox"]
        bb[0] = min(bb[0], x1)
        bb[1] = min(bb[1], y1)
        bb[2] = max(bb[2], x2)
        bb[3] = max(bb[3], y2)
    best_block = None
    best_score = 0
    for bnum, info in blocks.items():
        text = " ".join(info["texts"])
        numeric_count = sum(1 for w in info["words"] if normalize_number_token(w["text"], style='unknown') is not None)
        score = sum(1 for k in keywords if k in text)
        score += numeric_count
        # Bonus: penalize if looks like address-only
        if numeric_count < 2:
            score -= 3
        if score > best_score:
            best_score = score
            best_block = (info["bbox"], info["words"])
    if best_score > 0:
        return best_block
    return None

# ---------------------------
# Column detection via KMeans (cv2.kmeans)
# ---------------------------
def infer_column_centers(words: List[Dict[str,Any]], k_clusters=4) -> List[float]:
    x_mids = []
    for w in words:
        xmid = (w['bbox'][0] + w['bbox'][2]) / 2.0
        x_mids.append([float(xmid)])
    if len(x_mids) == 0:
        return []
    pts = np.array(x_mids, dtype=np.float32)
    K = min(max(2, k_clusters), len(pts))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    attempts = 5
    try:
        compactness, labels, centers = cv2.kmeans(pts, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
        centers = sorted([c[0] for c in centers])
        return centers
    except Exception:
        xs = sorted([p[0] for p in pts.tolist()])
        centers = []
        for i in range(K):
            q = int(len(xs) * (i+0.5)/K)
            centers.append(xs[max(0, min(len(xs)-1, q))])
        return centers

# ---------------------------
# Reconstruct rows inside a bbox using y-mid clustering
# ---------------------------
def group_words_to_rows(words: List[Dict[str,Any]], row_tolerance=18) -> List[List[Dict[str,Any]]]:
    items = []
    for w in words:
        x_mid = (w['bbox'][0] + w['bbox'][2]) / 2.0
        y_mid = (w['bbox'][1] + w['bbox'][3]) / 2.0
        w2 = w.copy()
        w2['_xmid'] = x_mid
        w2['_ymid'] = y_mid
        items.append(w2)
    if not items:
        return []
    items.sort(key=lambda x: (x['_ymid'], x['_xmid']))
    rows = []
    current = [items[0]]
    ref = items[0]['_ymid']
    for it in items[1:]:
        if abs(it['_ymid'] - ref) <= row_tolerance:
            current.append(it)
        else:
            current.sort(key=lambda x: x['_xmid'])
            rows.append(current)
            current = [it]
            ref = it['_ymid']
    if current:
        current.sort(key=lambda x: x['_xmid'])
        rows.append(current)
    return rows

# ---------------------------
# Map word -> nearest center (column index)
# ---------------------------
def assign_cols_by_centers(row_words: List[Dict[str,Any]], centers: List[float]) -> Dict[int, List[Dict[str,Any]]]:
    col_map = {i: [] for i in range(len(centers))}
    for w in row_words:
        xmid = (w['bbox'][0] + w['bbox'][2]) / 2.0
        diffs = [abs(xmid - c) for c in centers]
        idx = int(np.argmin(diffs))
        col_map[idx].append(w)
    return col_map

# ---------------------------
# Strict row validator and extractor (now accepts page_style)
# ---------------------------
def extract_line_item_from_row(row: List[Dict[str,Any]], centers: List[float], page_style: str = 'unknown') -> Dict[str,Any]:
    def has_header_keyword(s):
        s = (s or "").lower()
        return any(k in s for k in ['subtotal', 'total', 'tax', 'invoice', 'payment', 'description', 'price', 'qty'])
    row_text = " ".join([w['text'] for w in row]).strip()
    if has_header_keyword(row_text) and not re.search(r'\d', row_text):
        return None
    numeric_tokens = []
    for w in row:
        val = normalize_number_token(w['text'], style=page_style)
        if val is not None:
            x_mid = (w['bbox'][0] + w['bbox'][2]) / 2.0
            numeric_tokens.append((w, val, x_mid))
    if not numeric_tokens:
        return None
    numeric_tokens_sorted_by_x = sorted(numeric_tokens, key=lambda t: t[2])
    w_amount, amount_val, x_amount = numeric_tokens_sorted_by_x[-1]

    # --- NEW: skip rows where rightmost numeric looks like a date or year ---
    try:
        amt_int = int(round(amount_val))
        if 1900 <= amt_int <= 2100:
            return None
    except Exception:
        pass
    if re.match(r'^\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}$', str(w_amount.get('text', ''))):
        return None
    # ------------------------------------------------------------------------

    quantity = None
    for wtok, v, xm in numeric_tokens_sorted_by_x:
        if 0 < v <= 50 and abs(round(v) - v) < 1e-6:
            quantity = int(round(v))
            break
    unit_price = None
    candidates = [t for t in numeric_tokens_sorted_by_x if t[2] < x_amount]
    if candidates:
        cand = candidates[-1]
        unit_price = float(cand[1])
    if unit_price is None and quantity and amount_val is not None:
        if quantity != 0:
            unit_price = amount_val / quantity
    if amount_val is None:
        return None
    if amount_val <= 0 or amount_val > 1e7:
        return None
    if unit_price is not None and (unit_price <= 0 or unit_price > 1e7):
        unit_price = None
    used_word_ids = set()
    used_word_ids.add(id(w_amount))
    if quantity is not None:
        for wtok, v, xm in numeric_tokens_sorted_by_x:
            if 0 < v <= 50 and abs(round(v) - v) < 1e-6:
                used_word_ids.add(id(wtok))
                break
    if unit_price is not None:
        for wtok, v, xm in numeric_tokens_sorted_by_x:
            if abs(float(v) - float(unit_price)) < 1e-6 and id(wtok) not in used_word_ids:
                used_word_ids.add(id(wtok))
                break
    name_parts = []
    for w in row:
        if id(w) in used_word_ids:
            continue
        txt = w['text'].strip()
        if not txt:
            continue
        if re.fullmatch(r'[\d\.,\-]+', txt):
            continue
        name_parts.append(txt)
    item_name = " ".join(name_parts).strip()

    # --- NEW: strip duplicate currency tokens and collapse whitespace ---
    # Remove repeated occurrences like "$10,00 $10,00" -> keep single "$10,00"
    item_name = re.sub(r'(?:\$\s?[\d\.,]+\s*){2,}', lambda m: m.group(0).strip().split()[0], item_name)
    item_name = re.sub(r'\s{2,}', ' ', item_name).strip()
    # ---------------------------------------------------------------------

    if not re.search(r'[A-Za-z]', item_name):
        return None
    return {
        "item_name": item_name,
        "quantity": quantity if quantity is not None else None,
        "unit_price": float(unit_price) if unit_price is not None else None,
        "amount": float(amount_val)
    }


# ---------------------------
# High-level table extractor (glues everything) --- accepts page_style
# ---------------------------
def extract_table_items_with_kmeans(raw_ocr_data: List[Dict[str,Any]], fallback_bbox: Tuple[int,int,int,int]=None, page_style: str = 'unknown') -> List[Dict[str,Any]]:
    found = find_table_block(raw_ocr_data)
    if found:
        block_bbox, block_words = found
    else:
        if fallback_bbox:
            x1,y1,x2,y2 = fallback_bbox
            block_words = [w for w in raw_ocr_data if (w['bbox'][0] >= x1 and w['bbox'][2] <= x2 and w['bbox'][1] >= y1 and w['bbox'][3] <= y2)]
            block_bbox = list(fallback_bbox)
        else:
            block_words = raw_ocr_data
            block_bbox = [0,0,1000000,1000000]
    if not block_words:
        return []
    centers = infer_column_centers(block_words, k_clusters=4)
    if not centers:
        x1,x2 = block_bbox[0], block_bbox[2]
        width = x2 - x1
        centers = [x1 + width*(i+0.5)/4.0 for i in range(4)]
    centers = sorted(centers)
    rows = group_words_to_rows(block_words, row_tolerance=18)
    items = []
    for row in rows:
        item = extract_line_item_from_row(row, centers, page_style=page_style)
        if item:
            items.append(item)
    return items

# ---------------------------
# Postprocess (clean names, infer qty=1 as before)
# ---------------------------
def postprocess_line_items(line_items):
    cleaned = []
    prev = None
    for item in line_items:
        name = (item.get("item_name") or "").strip()
        lname = name.lower()
        if any(k in lname for k in ['subtotal', 'tax', 'total', 'balance due', 'grand total']):
            row_type = 'total' if 'total' in lname else ('tax' if 'tax' in lname else 'subtotal')
            item['row_type'] = row_type
            if not item.get('amount'):
                continue
            cleaned.append(item)
            prev = item
            continue
        if not item.get('amount') or not re.search(r'[A-Za-z]', name):
            continue
        if item.get('quantity') is None:
            if item.get('unit_price') and item.get('amount'):
                if abs(item['unit_price'] - item['amount']) < 1e-2:
                    item['quantity'] = 1
            else:
                item['quantity'] = 1
        if (item.get('unit_price') is None or item.get('unit_price') == 0) and item.get('quantity') and item.get('amount'):
            try:
                up = item['amount'] / item['quantity']
                if 0 < up < 1e6:
                    item['unit_price'] = round(up, 2)
            except Exception:
                item['unit_price'] = None
        if item.get('unit_price') and (item['unit_price'] <= 0 or item['unit_price'] > 1e7):
            item['unit_price'] = None
        item['item_name'] = re.sub(r'[\|\u2014]+', ' ', name).strip()
        cleaned.append(item)
        prev = item
    return cleaned

# ---------------------------
# Integration into your pipeline (detect per-page style)
# ---------------------------
def process_document_kmeans(input_path, out_dir):
    ensure_dir(out_dir)
    pages = pdf_to_images(input_path)
    results = []
    all_line_items = []
    for i, img in enumerate(tqdm(pages, desc="Processing pages")):
        page_dir = os.path.join(out_dir, f"page_{i+1:03d}")
        ensure_dir(page_dir)
        deskewed, angle = deskew(img)
        denoised = remove_noise(deskewed)
        regions = segment_page(denoised)
        original_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        raw_ocr_data = run_tesseract_layout_ocr(denoised, numeric_conf_thresh=25)
        # persist raw ocr
        with open(os.path.join(page_dir, "raw_ocr_data.json"), "w") as f:
            json.dump(raw_ocr_data, f, indent=2)
        # detect page numeric style and pass to extractor
        page_style = detect_and_set_page_style(raw_ocr_data)
        _, y1, _, y2 = regions['body']
        fallback_bbox = (0, y1, denoised.shape[1], y2)
        items = extract_table_items_with_kmeans(raw_ocr_data, fallback_bbox=fallback_bbox, page_style=page_style)
        items = postprocess_line_items(items)
        all_line_items.extend(items)
        with open(os.path.join(page_dir, "line_items_kmeans.json"), "w") as f:
            json.dump(items, f, indent=2)
        # other checks (reuse your functions if present)
        whiteners = detect_whiteners(denoised) if 'detect_whiteners' in globals() else []
        fraud_ocr_checks = detect_tampered_numbers(raw_ocr_data) if 'detect_tampered_numbers' in globals() else {}
        numeric_patch_detected = detect_numeric_patches(original_gray, raw_ocr_data) if 'detect_numeric_patches' in globals() else False
        sample_text = " ".join([d['text'] for d in raw_ocr_data])
        lang = detect(sample_text[:500]) if sample_text else "unknown"
        cv2.imwrite(os.path.join(page_dir, "deskewed.jpg"), deskewed)
        cv2.imwrite(os.path.join(page_dir, "denoised.jpg"), denoised)
        summary = {
            "page": i+1,
            "skew_angle": angle,
            "language": lang,
            "table_count": 1 if items else 0,
            "fraud": {
                "whiteners": len(whiteners),
                "suspicious_numbers": len(fraud_ocr_checks.get("suspicious_numbers", [])),
                "inconsistent_fonts": bool(fraud_ocr_checks.get("inconsistent_fonts_detected", False)),
                "numeric_patch_detected": bool(numeric_patch_detected)
            },
            "line_items_extracted": len(items)
        }
        with open(os.path.join(page_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        results.append(summary)
    with open(os.path.join(out_dir, "run_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results, all_line_items

# ---------------------------
# CLI wrapper
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Path to input PDF or image")
    parser.add_argument("-o", "--out", required=True, help="Output directory")
    args = parser.parse_args()
    results, items = process_document_kmeans(args.input, args.out)
    print(json.dumps(results, indent=2))
    print("Extracted items:", len(items))
