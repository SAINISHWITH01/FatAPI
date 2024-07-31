from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
from transformers import pipeline
import fitz  # PyMuPDF
from docx import Document

# Create FastAPI app
app = FastAPI()

# Directory to save uploaded files
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load a pre-trained model for summarization
summarizer = pipeline("summarization")

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Document Summarizer"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as f:
        f.write(file.file.read())
    return {"info": f"file '{file.filename}' saved at '{file_location}'"}

@app.post("/summarize")
async def summarize_file(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as f:
        f.write(file.file.read())

    file_extension = os.path.splitext(file_location)[1].lower()
    text = ""

    if file_extension == ".pdf":
        doc = fitz.open(file_location)
        for page in doc:
            text += page.get_text()
    elif file_extension == ".docx":
        doc = Document(file_location)
        for para in doc.paragraphs:
            text += para.text
    elif file_extension == ".txt":
        with open(file_location, "r") as f:
            text = f.read()
    else:
        return JSONResponse(content={"error": "Unsupported file type"}, status_code=400)

    summary = summarizer(text, max_length=100, min_length=30, do_sample=False)

    return JSONResponse(content={"summary": summary[0]['summary_text']})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
