import logging
from typing import Dict
import os
from google import genai

logger = logging.getLogger(__name__)

API_KEY = os.environ.get("API_KEY", "")
client = genai.Client(api_key=API_KEY)

def generate_detailed_report(student_answers: Dict[str, str],
                             correct_answers: Dict[str, str],
                             similarity_scores: Dict[str, float]) -> str:
    
    prompt = (
        "Please provide a detailed report comparing the student's answers to the correct answers. "
        "Include the cosine similarity scores for each question, and explain why marks were reduced "
        "if the similarity score is below an acceptable threshold. Provide constructive feedback for improvement. "
        "The report should be clear and easy to understand for the student. Use simple language and avoid technical jargon. "
        f"Student Answers: {student_answers}. "
        f"Correct Answers: {correct_answers}. "
        f"Similarity Scores: {similarity_scores}."
    )
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        logger.error(f"Gemini API call failed: {str(e)}")
        return f"Error generating detailed report: {str(e)}"
