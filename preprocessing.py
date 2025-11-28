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
from typing import List, Dict, Any

# ---------------------------
# Utilities
# ---------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# ---------------------------
# PDF → Image
# ---------------------------
def pdf_to_images(pdf_path, dpi=300):
    # NOTE: Set your poppler_bin path here or configure the environment variable
    # Ensure this path is correct for your system
    #poppler_bin = r"D:\web development\poppler\poppler-25.11.0\Library\bin" 
    if pdf_path.lower().endswith('.pdf'):
        pages = convert_from_path(
            pdf_path,
            dpi=dpi,
            #poppler_path=poppler_bin
        )
    else:
        pages = [Image.open(pdf_path)]

    imgs = []
    for p in pages:
        rgb = p.convert('RGB')
        arr = np.array(rgb)[:, :, ::-1].copy()
        imgs.append(arr)
    return imgs

# ---------------------------
# Skew Correction
# ---------------------------
def deskew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100,
                            minLineLength=200, maxLineGap=20)
    angles = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angles.append(angle)

    if not angles:
        return image, 0.0

    median_angle = np.median(angles)
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated, round(float(median_angle), 2)

# ---------------------------
# Noise + Shadow Removal (Phase 1)
# ---------------------------
def remove_noise(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Adaptive Thresholding for clean binary image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    denoised = cv2.adaptiveThreshold(
        enhanced, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        11, 
        2 
    )
    return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)

# ---------------------------
# Page Segmentation (Phase 1)
# ---------------------------
def segment_page(image):
    h, w = image.shape[:2]
    header = (0, 0, w, int(0.12 * h))
    body = (0, int(0.12 * h), w, int(0.88 * h))
    footer = (0, int(0.88 * h), w, h)
    return {"header": header, "body": body, "footer": footer}

# ---------------------------
# Table Detection (Phase 1)
# ---------------------------
def detect_tables(image, out_debug=None):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))

    horiz = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_h)
    vert = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_v)

    mask = cv2.add(horiz, vert)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    boxes = []
    H, W = image.shape[:2]
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w * h > 0.01 * H * W and w > 0.2 * W:
            boxes.append((x, y, x + w, y + h))

    if out_debug:
        vis = image.copy()
        for b in boxes:
            cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
        cv2.imwrite(out_debug, vis)

    return boxes

# ---------------------------
# Language Detection
# ---------------------------
def detect_language(text):
    try:
        # Use only a small chunk of text for speed
        return detect(text[:500]) 
    except LangDetectException:
        return "unknown"

# ---------------------------
# Fraud Detection (Phase 1 & 5)
# ---------------------------
def detect_whiteners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    H, W = image.shape[:2]
    return [
        cv2.boundingRect(c)
        for c in cnts
        if cv2.contourArea(c) > 0.002 * H * W
    ]

def detect_tampered_numbers(raw_ocr_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Detects low-confidence numbers and flags blocks with potential font/DPI inconsistency 
    using word area / character count as a proxy.
    """
    suspicious_numbers = []
    area_per_char_ratios = []
    
    for word_data in raw_ocr_data:
        text = word_data.get("text", "")
        confidence = word_data.get("confidence", 0.0)
        bbox = word_data.get("bbox", [0, 0, 0, 0])
        
        # 1. Low Confidence Number Check
        if text.strip().isdigit():
            if confidence < 60:
                suspicious_numbers.append(f"{text} (Conf: {confidence:.1f})")
                
        # 2. DPI/Font Inconsistency Check (Proxy: Word Area / Char Count)
        if len(text) > 2:
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            # FIX: Filter out likely headings (height > 60 pixels) to prevent false positives
            if h > 60: 
                continue
            
            area = w * h
            char_count = len(text)
            
            if char_count > 0 and area > 0:
                area_per_char_ratios.append(area / char_count)

    # Analyze ratios: Count outliers outside the 3-sigma rule
    inconsistent_fonts = False
    if area_per_char_ratios:
        ratios = np.array(area_per_char_ratios)
        # Ensure mean and std are standard Python floats before comparison, though generally safe.
        mean_ratio = np.mean(ratios) 
        std_ratio = np.std(ratios)
        
        # inconsistent_count will be a NumPy integer (np.int64), needs casting for safety later
        inconsistent_count = np.sum(np.abs(ratios - mean_ratio) > 3 * std_ratio) 
        
        # Flag if more than 1% of words are inconsistent
        # inconsistent_fonts here is a NumPy bool_
        inconsistent_fonts = inconsistent_count > (len(ratios) * 0.01)

    return {
        "suspicious_numbers": suspicious_numbers,
        "inconsistent_fonts_detected": inconsistent_fonts
    }

def detect_numeric_patches(original_gray_image: np.array, raw_ocr_data: List[Dict[str, Any]]) -> bool:
    """Checks for background color inconsistencies around numeric fields."""
    
    H, W = original_gray_image.shape
    PATCH_THRESHOLD = 20 # Mean pixel difference to flag as suspicious

    for word_data in raw_ocr_data:
        text = word_data.get("text", "")
        
        # Check for numbers or currency amounts
        if text.strip().isdigit() or any(c in text for c in ['$', '.', ',']):
            x1, y1, x2, y2 = word_data.get("bbox", [0, 0, 0, 0])
            
            # Define inner patch area (the number itself)
            inner_patch = original_gray_image[y1:y2, x1:x2]
            
            # Define a small outer margin (5-pixel border)
            x1_margin = max(0, x1 - 5)
            y1_margin = max(0, y1 - 5)
            x2_margin = min(W, x2 + 5)
            y2_margin = min(H, y2 + 5)
            
            margin_area = original_gray_image[y1_margin:y2_margin, x1_margin:x2_margin]
            
            if margin_area.size == 0 or inner_patch.size == 0:
                continue
                
            # These means are NumPy floats
            mean_margin = np.mean(margin_area) 
            mean_inner = np.mean(inner_patch)
            
            if abs(mean_margin - mean_inner) > PATCH_THRESHOLD:
                # Returns Python bool True
                return True
                
    # Returns Python bool False
    return False

# ---------------------------
# OCR Layer (Phase 2)
# ---------------------------
def run_tesseract_layout_ocr(denoised_image):
    """Performs layout-aware OCR using Tesseract to extract text, bounding boxes, and confidence."""
    
    pil_img = Image.fromarray(cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB))
    data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT, config='--psm 6')

    structured_output = []
    n_boxes = len(data.get('level', []))
    
    for i in range(n_boxes):
        text = data.get('text', [''])[i].strip()
        confidence = float(data.get('conf', [0.0])[i])
        
        if text and confidence > 50: 
            # Note: Tesseract returns standard Python integers here, so casting is not strictly needed.
            x = data.get('left', [0])[i]
            y = data.get('top', [0])[i]
            w = data.get('width', [0])[i]
            h = data.get('height', [0])[i]
            
            bbox = [x, y, x + w, y + h]
            
            structured_output.append({
                "text": text,
                "confidence": confidence,
                "bbox": bbox,
                "page": data.get('page_num', [1])[i],
                "block_num": data.get('block_num', [1])[i]
            })
            
    return structured_output

# ---------------------------
# Line-item Extraction (Phase 3)
# ---------------------------
def reconstruct_table_rows(raw_ocr_data, table_bbox, row_tolerance=40):
    """Groups words into logical table rows based on vertical proximity, 
    returning full word objects including bboxes."""
    
    x1_table, y1_table, x2_table, y2_table = table_bbox
    table_words = []
    
    for word_data in raw_ocr_data:
        x_mid = (word_data['bbox'][0] + word_data['bbox'][2]) / 2
        y_mid = (word_data['bbox'][1] + word_data['bbox'][3]) / 2
        
        if (x1_table < x_mid < x2_table) and (y1_table < y_mid < y2_table):
            table_words.append(word_data)

    if not table_words:
        return []

    # Sort words primarily by Y and secondarily by X
    table_words.sort(key=lambda w: w['bbox'][1] + (w['bbox'][0] / 10000))

    rows = []
    current_row = []
    current_row_y_ref = table_words[0]['bbox'][1] 

    for word_data in table_words:
        word_y = word_data['bbox'][1]

        if abs(word_y - current_row_y_ref) < row_tolerance:
            current_row.append(word_data)
        else:
            if current_row:
                # IMPORTANT CHANGE: Append the list of dictionaries (word objects)
                rows.append(current_row) 

            current_row = [word_data]
            current_row_y_ref = word_y

    if current_row:
        # IMPORTANT CHANGE: Append the list of dictionaries (word objects)
        rows.append(current_row)

    return rows

def classify_table_columns(reconstructed_rows: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Classifies reconstructed rows into specific columns (Description, Qty, Price, Amount)."""
    
    line_items = []
    
    if len(reconstructed_rows) < 2:
        return []
    
    # 1. Determine table width for simple column partitioning
    # Use the first row to approximate the table's total width (min X to max X)
    all_x = [w['bbox'][0] for row in reconstructed_rows for w in row] + \
            [w['bbox'][2] for row in reconstructed_rows for w in row]
    
    if not all_x:
        return []

    min_x = min(all_x)
    max_x = max(all_x)
    table_width = max_x - min_x
    
    # Simple four-column partitioning (e.g., Qty, Desc, Price, Total)
    # This assumes the columns are roughly equally spaced. Adjust zone size if needed.
    zone_1_x_max = min_x + table_width * 0.15 # 0-15% (e.g., Quantity)
    zone_2_x_max = min_x + table_width * 0.70 # 15-70% (e.g., Description)
    zone_3_x_max = min_x + table_width * 0.85 # 70-85% (e.g., Unit Price)
    # Zone 4 is 85-100% (e.g., Amount)

    # 2. Process data rows (skipping assumed header)
    for row in reconstructed_rows[1:]:
        item_name_words = []
        quantity_words = []
        unit_price_words = []
        amount_words = []
        
        for word_data in row:
            text = word_data['text']
            x_mid = (word_data['bbox'][0] + word_data['bbox'][2]) / 2
            
            # Categorize word based on horizontal position
            if x_mid < zone_1_x_max:
                quantity_words.append(text)
            elif x_mid < zone_2_x_max:
                item_name_words.append(text)
            elif x_mid < zone_3_x_max:
                unit_price_words.append(text)
            else:
                amount_words.append(text)
                
        # 3. Extract final data
        
        def extract_numeric(words):
            """Finds the first clear numeric or currency value."""
            if not words:
                return None
            
            # Simple regex to find money format ($, decimal, comma) or pure numbers
            text_str = " ".join(words).replace('$', '').replace(',', '')
            match = re.search(r'\d+(\.\d{1,2})?', text_str)
            return float(match.group(0)) if match else None

        # Clean up item name - remove stray numbers that should be Qty/Price
        item_name = " ".join(item_name_words).strip()
        
        line_items.append({
            "item_name": item_name,
            "quantity": extract_numeric(quantity_words),
            "unit_price": extract_numeric(unit_price_words),
            "amount": extract_numeric(amount_words)
        })
            
    return line_items

# ---------------------------
# Main Pipeline
# ---------------------------
def process_document(input_path, out_dir):
    ensure_dir(out_dir)
    pages = pdf_to_images(input_path)

    results = []
    all_line_items = [] # Initialize list to collect all line items across pages

    for i, img in enumerate(tqdm(pages, desc="Processing pages")):
        page_dir = os.path.join(out_dir, f"page_{i+1:03d}")
        ensure_dir(page_dir)

        # PHASE 1: Preprocessing
        # angle is a float, which is JSON serializable
        deskewed, angle = deskew(img) 
        denoised = remove_noise(deskewed)
        regions = segment_page(denoised)
        tables = detect_tables(denoised, out_debug=os.path.join(page_dir, "tables.jpg"))
        original_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        
        # PHASE 2: OCR Layer
        raw_ocr_data = run_tesseract_layout_ocr(denoised)
        ocr_output_path = os.path.join(page_dir, "raw_ocr_data.json")
        with open(ocr_output_path, "w") as f:
            json.dump(raw_ocr_data, f, indent=2)
        
        # PHASE 3: Line Item Extraction
        # FIX: Bypass unreliable line-based table detection for bbox selection.
        # Force the extraction to use the entire content body region.
        _, y1, _, y2 = regions['body']
        main_table_bbox = (0, y1, denoised.shape[1], y2)
        
        reconstructed_rows = reconstruct_table_rows(raw_ocr_data, main_table_bbox)
        rows_output_path = os.path.join(page_dir, "reconstructed_rows.json")
        with open(rows_output_path, "w") as f:
            json.dump(reconstructed_rows, f, indent=2)
        
        line_items = classify_table_columns(reconstructed_rows)
        
        # Collect line items from this page
        all_line_items.extend(line_items) 
        
        line_items_output_path = os.path.join(page_dir, "line_items_classified.json")
        with open(line_items_output_path, "w") as f:
            json.dump(line_items, f, indent=2)
            
        # PHASE 5: Advanced Fraud Detection
        whiteners = detect_whiteners(denoised)
        fraud_ocr_checks = detect_tampered_numbers(raw_ocr_data)
        numeric_patch_detected = detect_numeric_patches(original_gray, raw_ocr_data)
        
        # Summary Metrics
        sample_text = " ".join([d['text'] for d in raw_ocr_data])
        lang = detect_language(sample_text)

        cv2.imwrite(os.path.join(page_dir, "deskewed.jpg"), deskewed)
        cv2.imwrite(os.path.join(page_dir, "denoised.jpg"), denoised)

        summary = {
            "page": i + 1,
            "skew_angle": angle,
            "language": lang,
            "table_count": len(tables),
            "fraud": {
                # len() returns a standard Python int
                "whiteners": len(whiteners), 
                "suspicious_numbers": len(fraud_ocr_checks["suspicious_numbers"]),
                # FIX: Explicitly cast NumPy bool_ to standard Python bool
                "inconsistent_fonts": bool(fraud_ocr_checks["inconsistent_fonts_detected"]), 
                "numeric_patch_detected": bool(numeric_patch_detected)
            },
            "line_items_extracted": len(line_items)
        }

        with open(os.path.join(page_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        results.append(summary)

    with open(os.path.join(out_dir, "run_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
        
    # Modified return to send BOTH summary and extracted data
    return results, all_line_items 


# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Path to the input PDF or image file.")
    parser.add_argument("-o", "--out", required=True, help="Path to the output directory.")
    args = parser.parse_args()

    # Note: CLI execution doesn't use the second return value
    process_document(args.input, args.out) 
    print("✅ Full preprocessing and extraction framework complete.")