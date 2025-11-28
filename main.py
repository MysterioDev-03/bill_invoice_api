import os
import shutil
import tempfile
import time
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

from preprocessing import process_document_kmeans, ensure_dir

BASE_OUTPUT_DIR = "data/outputs"
ensure_dir(BASE_OUTPUT_DIR)

app = FastAPI(
    title="HackRx Bill Extraction API",
    description="Extracts bill line items in the HackRx required format"
)

# -------------------------
# Request Body
# -------------------------
class ExtractRequest(BaseModel):
    document: str   # URL of document


# -------------------------
# Response Schema
# -------------------------
class BillItem(BaseModel):
    item_name: str
    item_amount: float
    item_rate: float
    item_quantity: float


class PageLineItems(BaseModel):
    page_no: str
    page_type: str
    bill_items: List[BillItem]


class ExtractResponse(BaseModel):
    is_success: bool
    token_usage: Dict[str, int]
    data: Dict[str, Any]


# ----------------------------------------------------
# HackRx API: POST /extract-bill-data
# ----------------------------------------------------
@app.post("/extract-bill-data", response_model=ExtractResponse)
async def extract_bill_data(req: ExtractRequest):

    # Create temp directory
    with tempfile.TemporaryDirectory() as temp_dir:

        try:
            # Download document from URL
            file_url = req.document
            file_name = file_url.split("/")[-1].split("?")[0]
            input_path = os.path.join(temp_dir, file_name)

            r = requests.get(file_url, stream=True)
            if r.status_code != 200:
                raise Exception("Failed to download document.")

            with open(input_path, "wb") as f:
                shutil.copyfileobj(r.raw, f)

            # Permanent output dir
            timestamp = int(time.time())
            output_dir = os.path.join(BASE_OUTPUT_DIR, f"{file_name}_{timestamp}")
            ensure_dir(output_dir)

            # Run your ML pipeline
            run_summary, extracted_items = process_document_kmeans(input_path, output_dir)

            # -----------------------------
            # Convert your output → HackRx format
            # -----------------------------
            pagewise_output = []
            total_items = 0

            for page in run_summary:
                page_items = []
                for it in extracted_items:
                    page_items.append({
                        "item_name": it.item_name,
                        "item_amount": it.amount or 0.0,
                        "item_rate": it.unit_price or 0.0,
                        "item_quantity": it.quantity or 0.0
                    })
                    total_items += 1

                pagewise_output.append({
                    "page_no": str(page.page),
                    "page_type": "Bill Detail",     # If you have logic, replace
                    "bill_items": page_items
                })

            # -----------------------------
            # Final response
            # -----------------------------
            return {
                "is_success": True,
                "token_usage": {
                    "total_tokens": 0,
                    "input_tokens": 0,
                    "output_tokens": 0
                },
                "data": {
                    "pagewise_line_items": pagewise_output,
                    "total_item_count": total_items
                }
            }

        except Exception as e:
            print("ERROR:", e)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process document: {str(e)}"
            )


@app.get("/")
def health():
    return {"status": "ok"}
