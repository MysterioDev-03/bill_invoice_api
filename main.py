@app.post("/extract-bill-data")
async def extract_bill_data(url: str):
    """
    Accepts URL directly: /extract-bill-data?url=https://...
    """

    import os, tempfile, shutil, time, requests
    from preprocessing import process_document_kmeans, ensure_dir

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Download file
            file_name = url.split("/")[-1].split("?")[0]
            input_path = os.path.join(temp_dir, file_name)

            r = requests.get(url, stream=True)
            if r.status_code != 200:
                raise Exception("Failed to download file")

            with open(input_path, "wb") as f:
                shutil.copyfileobj(r.raw, f)

            # Output folder
            timestamp = int(time.time())
            output_dir = os.path.join("data/outputs", f"{file_name}_{timestamp}")
            ensure_dir(output_dir)

            # Run pipeline
            run_summary, extracted_items = process_document_kmeans(input_path, output_dir)

            # Format exactly as HackRx wants
            pagewise = []
            total_count = 0

            for page in run_summary:
                items = []
                for it in extracted_items:
                    items.append({
                        "item_name": it.item_name,
                        "item_amount": it.amount or 0.0,
                        "item_rate": it.unit_price or 0.0,
                        "item_quantity": it.quantity or 0.0
                    })
                    total_count += 1

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
                    "total_item_count": total_count
                }
            }

        except Exception as e:
            raise HTTPException(500, f"Processing failed: {e}")
