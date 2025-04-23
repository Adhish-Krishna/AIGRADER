import os
import io
import json
from fastapi import UploadFile, HTTPException
from pdf2image import convert_from_bytes
from google.genai import types
from model import client

model_name = os.getenv("MODEL_NAME")

async def extract_text_from_upload(upload_file: UploadFile) -> str:
    """
    Extract text from uploaded PDF or image using Google Gen AI SDK.
    """
    content = await upload_file.read()
    ext = os.path.splitext(upload_file.filename)[1].lower()
    extracted_texts = []
    prompt = (
        "Extract all text content from this document/image. "
        "Format it clearly and preserve any important formatting."
    )

    if ext == ".pdf":
        pages = convert_from_bytes(content)
        for i, page in enumerate(pages):
            try:
                buf = io.BytesIO()
                page.save(buf, format="PNG")
                image_bytes = buf.getvalue()

                parts = [
                    types.Part.from_text(text=prompt),
                    types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                ]

                response = await client.aio.models.generate_content(
                    model=model_name,
                    contents=types.Content(parts=parts),
                )
                extracted_texts.append(response.text)
            except Exception as e:
                print(f"Error processing PDF page {i+1}: {e}")
                extracted_texts.append(f"[Error extracting text from page {i+1}]")

    elif ext in [".jpg", ".jpeg", ".png", ".webp"]:
        try:
            mime_type = f"image/{ext[1:]}"
            if ext == ".jpg":
                mime_type = "image/jpeg"

            parts = [
                types.Part.from_text(text=prompt),
                types.Part.from_bytes(data=content, mime_type=mime_type),
            ]

            response = await client.aio.models.generate_content(
                model=model_name,
                contents=types.Content(parts=parts),
            )
            extracted_texts.append(response.text)
        except Exception as e:
            error_msg = f"Error processing image {upload_file.filename}: {e}"
            print(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    return "\n".join(extracted_texts)

async def call_grading_model(prompt: str) -> dict:
    """
    Call Google Gen AI SDK to grade the exam based on the prompt.
    Returns JSON-decoded grading result.
    """
    try:
        response = await client.aio.models.generate_content(
            model=model_name,
            contents=types.Content(parts=[types.Part.from_text(text=prompt)]),
            config={"response_mime_type": "application/json"},
        )
        # Parse JSON result
        return json.loads(response.text)
    except json.JSONDecodeError:
        error_msg = f"Invalid JSON from grading model: {response.text}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    except Exception as e:
        error_msg = f"Error calling grading model: {e}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
