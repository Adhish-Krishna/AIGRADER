import logging
from typing import Dict, List, Tuple, Any
from app.database.db import get_answer_keys
from app.services.nlp_utils import compute_cosine_similarity  # NLP-based similarity function
from app.services.llm_eval import generate_detailed_report  # Gemini API based detailed report

logger = logging.getLogger(__name__)

def score_answers(student_answers: List[Dict[str, Any]], subject_id: str) -> Tuple[float, str]:
    
    logger.info(f"Scoring answers for subject {subject_id}")
    
    try:
        # Get all answer keys
        answer_keys = get_answer_keys()
        
        # Find the answer key for this subject
        subject_key = None
        for key_data in answer_keys.values():
            if key_data.get("subject_id") == subject_id:
                subject_key = key_data
                break
        
        if not subject_key:
            logger.warning(f"No answer key found for subject {subject_id}")
            return 0, "No answer key found for this subject."
        
        # Get the correct answers from the answer key
        correct_answers = subject_key.get("answers", {})
        if not correct_answers:
            logger.warning(f"Answer key for subject {subject_id} has no answers")
            return 0, "Answer key has no answers defined."
        
        # Convert student answers to a dict for easier lookup by question number.
        student_answers_dict = {}
        for answer in student_answers:
            q_num = answer.get("question_number")
            if q_num is not None:
                student_answers_dict[str(q_num)] = answer.get("answer_text", "")
        
        total_questions = len(correct_answers)
        total_score = 0.0
        answer_scores = {}  # Store cosine similarity score per answer
        
        for q_num, correct_text in correct_answers.items():
            student_text = student_answers_dict.get(str(q_num), "")
            
            # Compute cosine similarity between the student answer and the correct answer.
            similarity = compute_cosine_similarity(student_text, correct_text)
            answer_scores[str(q_num)] = similarity
            
            # Assume each question contributes a maximum of 1 mark.
            total_score += similarity
        
        # Calculate final mark as an average percentage.
        final_percentage = (total_score / total_questions) * 100 if total_questions > 0 else 0
        
        logger.info(f"Final score for subject {subject_id}: {final_percentage:.2f}% based on cosine similarities.")
        
        # Generate a detailed report using the Gemini API.
        detailed_report = generate_detailed_report(
            student_answers=student_answers_dict,
            correct_answers=correct_answers,
            similarity_scores=answer_scores
        )
        
        return final_percentage, detailed_report
        
    except Exception as e:
        logger.error(f"Error scoring answers: {str(e)}")
        return 0, f"Error scoring answers: {str(e)}"
