import os
import shutil
import tempfile 
import time 
import re # <-- Added for the column classification logic
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any

# Assuming your processing logic is in preprocessing.py
# Make sure you have applied the column classification fixes 
# to your 'preprocessing.py' file as discussed previously!
from preprocessing import process_document_kmeans, ensure_dir

# ----------------------------------------------------
# 1. Initialization and Data Models
# ----------------------------------------------------
class LineItem(BaseModel):
    item_name: str
    quantity: float | None
    unit_price: float | None
    amount: float | None

class FraudMetrics(BaseModel):
    whiteners: int
    suspicious_numbers: int
    inconsistent_fonts: bool
    numeric_patch_detected: bool

class PageSummary(BaseModel):
    page: int
    skew_angle: float
    language: str
    table_count: int
    fraud: FraudMetrics
    line_items_extracted: int

class ProcessedDocument(BaseModel):
    status: str
    file_name: str
    results: List[PageSummary]
    extracted_data: List[LineItem] 

# Initialize FastAPI app
app = FastAPI(
    title="Invoice Processing and Fraud Detection API",
    description="A high-speed engine for bill processing using ML and heuristics."
)

# Define the base directory for all PERMANENT outputs (logs, final JSONs, debug images)
BASE_OUTPUT_DIR = "data/outputs" 
ensure_dir(BASE_OUTPUT_DIR) # Ensure this base directory exists at startup

# ----------------------------------------------------
# 2. API Endpoint
# ----------------------------------------------------

@app.post("/process_invoice", response_model=ProcessedDocument)
async def process_invoice_file(file: UploadFile = File(...)):
    """
    Accepts an invoice file (PDF or image), runs the full pipeline, 
    saves logs to a permanent folder, and returns the extracted data.
    """
    
    # 1. Use a Temporary Directory for safe, automatic cleanup
    with tempfile.TemporaryDirectory() as temp_dir:
        
        # Define the path for the uploaded input file inside the temporary folder
        input_path = os.path.join(temp_dir, file.filename)
        
        # Define the unique PERMANENT output directory for this run's logs/outputs
        timestamp = int(time.time())
        file_base_name = os.path.splitext(file.filename)[0]
        unique_folder_name = f"{file_base_name}_{timestamp}"
        output_dir = os.path.join(BASE_OUTPUT_DIR, unique_folder_name)
        
        try:
            # Save the uploaded file to the temporary location
            # Note: This is crucial as file.file (a SpooledTemporaryFile) is only good for one read.
            with open(input_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Ensure the permanent output directory exists before running the pipeline
            ensure_dir(output_dir)

            # Run the full processing pipeline
            run_summary, extracted_items = process_document(input_path, output_dir)
            
            # Format and return the final JSON response
            return {
                "status": "success",
                "file_name": file.filename,
                "results": run_summary,
                "extracted_data": extracted_items
            }

        except Exception as e:
            # Return a standard HTTP error response
            print(f"Error processing file: {e}")
            # Clean up the permanent directory if it was created but the run failed
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            
            raise HTTPException(
                status_code=500, 
                detail=f"An error occurred during processing: {str(e)}"
            )
        
    # Temporary directory is cleaned up automatically here.

# ----------------------------------------------------
# 3. Health Check Endpoint
# ----------------------------------------------------
@app.get("/")
def health_check():
    """A simple endpoint to confirm the API is running."""
    return {"status": "ok", "message": "Invoice Processor API is running."}