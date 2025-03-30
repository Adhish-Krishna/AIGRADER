import uvicorn
from fastapi import FastAPI, UploadFile, File, Request, Form, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
import logging
from typing import List, Optional
import shutil
from pathlib import Path
import time
import json
import uuid

# Fix the imports - we need to make sure the app module is importable
try:
    from app.config import (
        TEMPLATE_DIR, STATIC_DIR, UPLOADS_DIR, 
        API_TITLE, API_VERSION, HOST, PORT, DEBUG, MAX_UPLOAD_SIZE,
        ANSWER_KEYS_FILE
    )
    from app.services.image_service import parse_answer_sheet
    from app.services.scoring_service import score_answers
    from app.database.db import save_submission, get_submission, get_submissions, save_answer_key, get_answer_key
except ImportError:
    # When running directly, adjust Python path
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from app.config import (
        TEMPLATE_DIR, STATIC_DIR, UPLOADS_DIR, 
        API_TITLE, API_VERSION, HOST, PORT, DEBUG, MAX_UPLOAD_SIZE,
        ANSWER_KEYS_FILE
    )
    from app.services.image_service import parse_answer_sheet
    from app.services.scoring_service import score_answers
    from app.database.db import save_submission, get_submission, get_submissions, save_answer_key, get_answer_key

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("aigrader")

app = FastAPI(title=API_TITLE, version=API_VERSION)

# Set up templates and static files
templates = Jinja2Templates(directory=TEMPLATE_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Background task for processing submissions
async def process_submission_task(submission_id: str):
    """Background task to process a submission"""
    try:
        # Add a short delay to simulate processing time
        time.sleep(1)
        
        submission = get_submission(submission_id)
        if not submission:
            logger.error(f"Submission not found: {submission_id}")
            return
        
        # Parse the answer sheet using OCR
        answers = await parse_answer_sheet(submission["file_path"], is_answer_key=False)
        
        # Score the answers - remove await as it's not an async function
        score, feedback = score_answers(
            answers, 
            submission["subject_id"]
        )
        
        # Update the submission with the score
        save_submission(
            submission_id=submission_id,
            student_id=submission["student_id"],
            subject_id=submission["subject_id"],
            file_path=submission["file_path"],
            answers=answers,
            grades={"score": score, "feedback": feedback, "status": "completed"}
        )
        
        logger.info(f"Processed submission {submission_id} with score {score:.1f}")
        
    except Exception as e:
        logger.error(f"Error processing submission {submission_id}: {str(e)}")
        # Update the submission to show the error
        if 'submission' in locals():
            save_submission(
                submission_id=submission_id,
                student_id=submission["student_id"],
                subject_id=submission["subject_id"],
                file_path=submission["file_path"],
                answers=[],  # Empty answers list
                grades={"status": "error", "error_message": str(e)}
            )
        else:
            logger.error(f"Could not update submission {submission_id} with error status")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the home page with upload form"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload", response_class=HTMLResponse)
async def upload_file(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    subject_id: str = Form(...),
    student_id: str = Form(...),
):
    """Handle file upload and start grading process"""
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Only image files are allowed")
        
        # Create a unique filename
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{student_id}_{subject_id}_{int(time.time())}{file_extension}"
        file_location = os.path.join(UPLOADS_DIR, unique_filename)
        
        # Save the uploaded file
        with open(file_location, "wb") as f:
            file_content = await file.read()
            
            # Check file size
            if len(file_content) > MAX_UPLOAD_SIZE:
                raise HTTPException(status_code=400, detail=f"File size exceeds {MAX_UPLOAD_SIZE // 1024 // 1024}MB limit")
                
            f.write(file_content)
        
        # Generate a unique submission ID
        submission_id = str(uuid.uuid4())
        
        # Create initial empty answers list - will be filled in during processing
        empty_answers = []
        
        # Save submission to database with all required parameters
        success = save_submission(
            submission_id=submission_id,
            student_id=student_id,
            subject_id=subject_id,
            file_path=file_location,
            answers=empty_answers,  # Empty list initially
            grades={"status": "processing"}  # Add status to grades dictionary
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save submission to database")
        
        # Process the file in background
        background_tasks.add_task(process_submission_task, submission_id)
        
        return templates.TemplateResponse(
            "processing.html", 
            {"request": request, "submission_id": submission_id}
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@app.get("/status/{submission_id}", response_class=HTMLResponse)
async def check_status(request: Request, submission_id: str):
    """Check the status of a submission"""
    submission = get_submission(submission_id)
    
    if not submission:
        return templates.TemplateResponse(
            "error.html", 
            {"request": request, "error_message": f"Submission {submission_id} not found"}
        )
    
    # Extract status from the grades dictionary    
    grades = submission.get("grades", {})
    status = grades.get("status", "processing")
        
    if status == "completed":
        return templates.TemplateResponse(
            "result.html", 
            {
                "request": request, 
                "score": grades.get("score", 0),
                "feedback": grades.get("feedback", ""),
                "student_id": submission["student_id"],
                "subject_id": submission["subject_id"]
            }
        )
    elif status == "error":
        return templates.TemplateResponse(
            "error.html", 
            {
                "request": request,
                "error_message": grades.get("error_message", "Unknown error occurred")
            }
        )
    
    return templates.TemplateResponse(
        "processing.html", 
        {"request": request, "submission_id": submission_id}
    )

def format_submissions_for_template(submissions_dict):
    """
    Format submissions data for use in templates.
    Adds the ID as a field in each submission record.
    
    Args:
        submissions_dict: Dictionary of submissions with IDs as keys
        
    Returns:
        List of submission objects with ID included in each record
    """
    formatted_submissions = []
    for submission_id, submission_data in submissions_dict.items():
        # Add the ID to the submission data
        submission_with_id = submission_data.copy()
        submission_with_id['id'] = submission_id
        formatted_submissions.append(submission_with_id)
    return formatted_submissions

@app.get("/admin", response_class=HTMLResponse)
async def admin_panel(request: Request):
    """Admin panel for managing answer keys and viewing submissions"""
    # Format submissions to include the ID field
    raw_submissions = get_submissions()
    submissions = format_submissions_for_template(raw_submissions)
    
    answer_keys = get_all_answer_keys()
    return templates.TemplateResponse(
        "admin.html", 
        {"request": request, "submissions": submissions, "answer_keys": answer_keys}
    )

@app.post("/admin/answer-key", response_class=HTMLResponse)
async def upload_answer_key(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    subject_id: str = Form(...),
):
    """Upload answer key for a subject"""
    try:
        # Create a unique filename
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{subject_id}_{int(time.time())}{file_extension}"
        file_location = os.path.join(UPLOADS_DIR, f"answer_key_{unique_filename}")
        
        # Save the uploaded file
        with open(file_location, "wb") as f:
            file_content = await file.read()
            
            # Check file size
            if len(file_content) > MAX_UPLOAD_SIZE:
                raise HTTPException(status_code=400, detail=f"File size exceeds {MAX_UPLOAD_SIZE // 1024 // 1024}MB limit")
                
            f.write(file_content)
        
        # Process the answer key - extract raw text with question numbers
        answers_dict = await parse_answer_sheet(file_location, is_answer_key=True)
        
        # Generate a unique key ID
        key_id = str(uuid.uuid4())
        
        # Save answer key info to database
        save_answer_key(key_id, subject_id, file_location, answers_dict)
        
        # Format submissions for the template
        raw_submissions = get_submissions()
        submissions = format_submissions_for_template(raw_submissions)
        
        answer_keys = get_all_answer_keys()
        
        return templates.TemplateResponse(
            "admin.html", 
            {"request": request, "message": "Answer key uploaded successfully", "submissions": submissions, "answer_keys": answer_keys}
        )
    except Exception as e:
        logger.error(f"Error uploading answer key: {str(e)}")
        
        # Format submissions for the template
        raw_submissions = get_submissions()
        submissions = format_submissions_for_template(raw_submissions)
        
        answer_keys = get_all_answer_keys()
        
        return templates.TemplateResponse(
            "admin.html", 
            {"request": request, "error": f"Error uploading answer key: {str(e)}", "submissions": submissions, "answer_keys": answer_keys}
        )

@app.get("/process/{submission_id}")
async def process_submission(
    submission_id: str,
    background_tasks: BackgroundTasks
):
    """Process a submission (for background tasks or manual triggering)"""
    submission = get_submission(submission_id)
    
    if not submission:
        raise HTTPException(status_code=404, detail=f"Submission {submission_id} not found")
    
    # Add to background tasks
    background_tasks.add_task(process_submission_task, submission_id)
    
    return RedirectResponse(url=f"/status/{submission_id}", status_code=303)

def get_all_answer_keys():
    """Get all answer keys (helper function)"""
    try:
        if os.path.exists(ANSWER_KEYS_FILE):
            with open(ANSWER_KEYS_FILE, 'r') as f:
                answer_keys = json.load(f)
            return list(answer_keys.values())
        return []
    except Exception as e:
        logger.error(f"Error retrieving answer keys: {str(e)}")
        return []

@app.on_event("startup")
async def startup_event():
    """Startup event"""
    logger.info(f"Starting AI Answer Sheet Grader on http://{HOST}:{PORT}")
    logger.info(f"Debug mode: {DEBUG}")
    
    # Check if Google Cloud Vision credentials are set
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        logger.warning("GOOGLE_APPLICATION_CREDENTIALS environment variable not set. Vision API may not work correctly.")

if __name__ == "__main__":
    uvicorn.run("main:app", host=HOST, port=PORT, reload=DEBUG)
