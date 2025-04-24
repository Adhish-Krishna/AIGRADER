from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from api.utils import extract_text_from_upload, call_grading_model
from prompts.prompt import build_grading_prompt
from api.schemas import GradeResponse

app = FastAPI()

@app.post("/grade", response_model=GradeResponse)
async def grade(student_doc: UploadFile = File(...), answer_key: UploadFile = File(...)):
    try:
        student_text = await extract_text_from_upload(student_doc)
        answer_key_text = await extract_text_from_upload(answer_key)
        prompt = build_grading_prompt(student_text, answer_key_text)
        result = await call_grading_model(prompt)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))