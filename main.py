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
            # --- Download file ---
            file_name = url.split("/")[-1].split("?")[0]
            input_path = os.path.join(temp_dir, file_name)

            r = requests.get(url, stream=True)
            if r.status_code != 200:
                raise Exception("Failed to download file")

            with open(input_path, "wb") as f:
                shutil.copyfileobj(r.raw, f)

            # --- Output folder ---
            timestamp = int(time.time())
            out_dir = os.path.join(BASE_OUTPUT_DIR, f"{file_name}_{timestamp}")
            ensure_dir(out_dir)

            # --- Process document ---
            run_summary, extracted_items = process_document_kmeans(input_path, out_dir)

            # --- Convert to HackRx JSON format ---
            pagewise_output = []
            total_items = 0

            for page in run_summary:
                page_items = []

                for it in extracted_items:
                    page_items.append({
                        "item_name": it.get("item_name", ""),
                        "item_amount": float(it.get("amount", 0.0)),
                        "item_rate": float(it.get("unit_price", 0.0)),
                        "item_quantity": float(it.get("quantity", 0.0))
                    })
                    total_items += 1

                pagewise_output.append({
                    "page_no": str(page["page"]),
                    "page_type": "Bill Detail",
                    "bill_items": page_items
                })

            # --- Final response ---
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
            raise HTTPException(status_code=500, detail=str(e))



@app.get("/")
def health():
    return {"status": "ok"}
