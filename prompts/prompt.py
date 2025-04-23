from string import Template

grading_prompt_template = Template(
    """You are an AI grader. The student exam sheet text:
${student_doc}

The answer key and marking schema:
${answer_key_schema}

The exam may contain questions in the following formats: MCQ, Fill-ups, True or False, 2 Marks Questions, 5 Marks Questions, 10 Marks Questions.

Your task:
- Evaluate each answer according to the marking schema.
- Assign marks to each question.
- Extract both the student's answer and expected answer for each question.
- Provide total score and detailed feedback per question.

Output strictly in JSON:
{
  "total_score": int,
  "max_possible_score": int,
  "results": [
    {
      "question_number": int,
      "question_type": str,
      "allocated_marks": int,
      "obtained_marks": int,
      "student_answer": str,
      "expected_answer": str,
      "feedback": str
    }
  ]
}
"""
)

def build_grading_prompt(student_doc: str, answer_key_schema: str) -> str:
    return grading_prompt_template.substitute(
        student_doc=student_doc,
        answer_key_schema=answer_key_schema
    )
