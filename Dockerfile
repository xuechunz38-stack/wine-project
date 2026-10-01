# Reproducible environment for the wine-quality analysis.
#   docker build -t wine-quality .
#   docker run --rm -v "$PWD/output:/app/output" wine-quality         # run the analysis
#   docker run --rm wine-quality pytest -q                              # run the test suite
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    WINE_OUTPUT_DIR=/app/output

WORKDIR /app

# Install dependencies first so this layer is cached when only code changes.
COPY requirements.txt requirements-dev.txt ./
RUN pip install -r requirements-dev.txt

COPY . .

# Run as a non-root user; outputs go to /app/output (mount a volume there).
RUN useradd --create-home appuser && mkdir -p /app/output && chown -R appuser /app
USER appuser

CMD ["python", "analysis.py"]
