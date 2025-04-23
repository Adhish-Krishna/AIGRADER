from pydantic import BaseModel, Field
from typing import List


class QuestionResult(BaseModel):
    question_number: int = Field(..., description="The number of the question")
    question_type: str = Field(..., description="Type of the question, e.g., MCQ, Fill-ups")
    allocated_marks: int = Field(..., description="Marks allocated for this question")
    obtained_marks: int = Field(..., description="Marks obtained by the student")
    student_answer: str = Field(..., description="The answer provided by the student")
    expected_answer: str = Field(..., description="The expected correct answer")
    feedback: str = Field(..., description="Feedback for the answer")


class GradeResponse(BaseModel):
    total_score: int = Field(..., description="Total marks obtained by the student")
    max_possible_score: int = Field(..., description="Maximum possible marks for the exam")
    results: List[QuestionResult] = Field(..., description="Detailed results per question")