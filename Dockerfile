FROM python:3.10

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python3", "src/ocr_pipeline.py", "--input", "med_patient#1.pdf", "--output_dir", "ocr_out/preprocessed_run_gpu_full", "--use_gpu", "--all_pages"]
