from fastapi import APIRouter, HTTPException
import logging

from app.database.db import get_submissions, get_submission, save_submission

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/submissions")
def list_all_submissions():
    """Get all student submissions"""
    try:
        submissions = get_submissions()
        return {"submissions": submissions}
    except Exception as e:
        logger.error(f"Error retrieving submissions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve submissions: {str(e)}")

@router.get("/submissions/{submission_id}")
def get_submission_by_id(submission_id: str):
    """Get a specific submission by ID"""
    try:
        submission = get_submission(submission_id)
        if not submission:
            raise HTTPException(status_code=404, detail="Submission not found")
        return {"submission": submission}
    except Exception as e:
        logger.error(f"Error retrieving submission {submission_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve submission: {str(e)}")

@router.post("/submissions")
def create_submission(submission_id: str, student_id: str, subject_id: str, file_path: str, answers: list, grades: dict = None):
    """Create a new submission"""
    try:
        success = save_submission(submission_id, student_id, subject_id, file_path, answers, grades)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save submission")
        return {"message": "Submission created successfully"}
    except Exception as e:
        logger.error(f"Error creating submission: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create submission: {str(e)}")