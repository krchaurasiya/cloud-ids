FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /srv

COPY app/requirements.txt app/requirements.txt
RUN pip install --no-cache-dir -r app/requirements.txt

# Application code + ALL model versions. Each Cloud Run revision picks one
# with MODEL_VERSION, so rollback is just a traffic switch between revisions.
COPY app/ app/
COPY models/ models/

RUN useradd --create-home appuser
USER appuser

EXPOSE 8080
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
