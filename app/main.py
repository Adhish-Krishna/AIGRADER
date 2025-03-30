import os
import uuid
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import Optional

from app.config import UPLOADS_DIR
from app.utils.file_utils import save_upload_file
from app.services.image_service import parse_answer_sheet
from app.database.db import save_answer_key, save_submission, get_submission, get_submissions

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aigrader")

app = FastAPI()

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    file_type: str = Form(...),
    subject_id: str = Form(...),
    student_id: Optional[str] = Form(None)
):
    """
    Handle file uploads for answer keys and student submissions
    """
    try:
        # Validate file type
        if file_type not in ["answer_key", "student_submission"]:
            return {"error": "Invalid file type. Must be 'answer_key' or 'student_submission'."}

        # Generate prefix based on file type
        prefix = f"answer_key_{subject_id}" if file_type == "answer_key" else f"submission_{subject_id}_{student_id}"
        
        # Save the uploaded file
        success, file_path, error_message = await save_upload_file(
            file=file, 
            directory=UPLOADS_DIR,
            prefix=prefix
        )
        
        if not success:
            return {"error": f"Failed to upload file: {error_message}"}
        
        # Process the file based on its type
        if file_type == "answer_key":
            # Parse answer key and save to database
            answers = await parse_answer_sheet(file_path, is_answer_key=True)
            
            success = save_answer_key(
                key_id=str(uuid.uuid4()),
                subject_id=subject_id,
                file_path=file_path,
                answers=answers
            )
            
            if not success:
                return {"error": "Failed to save answer key to database."}
                
            return {"message": "Answer key uploaded successfully.", "answers": answers}
        
        else:  # student submission
            # Parse student answers and save to database with all required parameters
            answers = await parse_answer_sheet(file_path, is_answer_key=False)
            
            submission_id = str(uuid.uuid4())
            success = save_submission(
                submission_id=submission_id,
                student_id=student_id,
                subject_id=subject_id,
                file_path=file_path,
                answers=answers
            )
            
            if not success:
                return {"error": "Failed to save submission to database."}
                
            return {"message": "Student submission uploaded successfully.", "answers": answers}
            
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        return {"error": f"An error occurred during file upload: {str(e)}"}