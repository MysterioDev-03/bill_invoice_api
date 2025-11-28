from fastapi import FastAPI, HTTPException
import os, tempfile, shutil, time, requests
from preprocessing import process_document_kmeans, ensure_dir

# ------------------------
# 1. Create FastAPI app FIRST
# ------------------------
app = FastAPI(
    title="HackRx Bill Extraction API",
    description="Extracts bill line items in the HackRx required format"
)

BASE_OUTPUT_DIR = "data/outputs"
ensure_dir(BASE_OUTPUT_DIR)

# ------------------------
# 2. NOW define the endpoint
# ------------------------
@app.post("/extract-bill-data")
async def extract_bill_data(url: str):
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Download the file
            file_name = url.split("/")[-1].split("?")[0]
            input_path = os.path.join(temp_dir, file_name)

            r = requests.get(url, stream=True)
            if r.status_code != 200:
                raise Exception("Failed to download file")

            with open(input_path, "wb") as f:
                shutil.copyfileobj(r.raw, f)

            # Build output dir
            timestamp = int(time.time())
            output_dir = os.path.join(BASE_OUTPUT_DIR, f"{file_name}_{timestamp}")
            ensure_dir(output_dir)

            run_summary, extracted_items = process_document_kmeans(input_path, output_dir)

            # Convert to HackRx response
            pagewise = []
            total = 0

            for page in run_summary:
                items = []
                for it in extracted_items:
                    items.append({
                        "item_name": it.item_name,
                        "item_amount": it.amount or 0.0,
                        "item_rate": it.unit_price or 0.0,
                        "item_quantity": it.quantity or 0.0
                    })
                    total += 1

                pagewise.append({
                    "page_no": str(page.page),
                    "page_type": "Bill Detail",
                    "bill_items": items
                })

            return {
                "is_success": True,
                "token_usage": {
                    "total_tokens": 0,
                    "input_tokens": 0,
                    "output_tokens": 0
                },
                "data": {
                    "pagewise_line_items": pagewise,
                    "total_item_count": total
                }
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def health():
    return {"status": "ok"}
