FROM python:3.11-slim@sha256:e41613d42d4891e4930f79523f93f81bbc7632584ec65e36ab055f41a800b41e
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --uid 10001 analyst
COPY *.py pytest.ini ./
COPY tests ./tests
COPY data ./data
RUN mkdir -p figures && chown -R analyst:analyst /app
USER analyst
CMD ["python", "analysis.py"]
