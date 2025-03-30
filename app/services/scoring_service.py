import logging
from typing import Dict, List, Tuple, Any
from app.database.db import get_answer_keys

logger = logging.getLogger(__name__)

def score_answers(student_answers: List[Dict[str, Any]], subject_id: str) -> Tuple[float, str]:
    """
    Score student answers against the answer key for the subject.
    
    Args:
        student_answers: List of dictionaries with question numbers and answers
        subject_id: ID of the subject
        
    Returns:
        Tuple of (score, feedback)
    """
    logger.info(f"Scoring answers for subject {subject_id}")
    
    try:
        # Get all answer keys
        answer_keys = get_answer_keys()
        
        # Find the answer key for this subject
        subject_key = None
        for key_id, key_data in answer_keys.items():
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
        
        # Convert student answers to the same format as the answer key
        student_answers_dict = {}
        for answer in student_answers:
            q_num = answer.get("question_number")
            if q_num:
                student_answers_dict[str(q_num)] = answer.get("answer_text", "")
        
        # Score each answer
        total_questions = len(correct_answers)
        correct_count = 0
        feedback_items = []
        
        for q_num, correct_text in correct_answers.items():
            student_text = student_answers_dict.get(str(q_num), "")
            
            # Simple comparison for now - could be improved with NLP techniques
            similarity = calculate_similarity(student_text, correct_text)
            
            if similarity >= 0.5:  # Arbitrary threshold
                correct_count += 1
                feedback_items.append(f"Q{q_num}: Correct ({similarity:.2f} similarity)")
            else:
                feedback_items.append(f"Q{q_num}: Incorrect ({similarity:.2f} similarity)")
        
        # Calculate score as percentage
        score = (correct_count / total_questions) * 100 if total_questions > 0 else 0
        
        # Compile feedback
        feedback = "\n".join(feedback_items)
        
        logger.info(f"Scored {correct_count}/{total_questions} correct for subject {subject_id}")
        return score, feedback
        
    except Exception as e:
        logger.error(f"Error scoring answers: {str(e)}")
        return 0, f"Error scoring answers: {str(e)}"

def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two text strings.
    This is a very simple implementation that could be improved with NLP techniques.
    
    Args:
        text1: First text string
        text2: Second text string
        
    Returns:
        Similarity score between 0 and 1
    """
    # Simple word overlap similarity for now
    if not text1 or not text2:
        return 0.0
    
    # Convert to lowercase and split into words
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    # Calculate Jaccard similarity
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0
