FROM python:3.11-slim

WORKDIR /app

# System deps for Pillow and torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download CLIP weights at build time so first request is fast
RUN python -c "\
from transformers import CLIPModel, CLIPProcessor; \
CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); \
CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32'); \
print('CLIP cached')"

# Copy application code
COPY app/ ./app/
COPY static/ ./static/

# artifacts/ is mounted as a volume — not copied into image
# so model files stay outside the image and can be updated without rebuild

ENV ARTIFACTS_DIR=/app/artifacts
ENV COUNTERFEIT_THRESHOLD=0.5

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
