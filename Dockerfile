# Use an official Python runtime as a parent image
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app/
RUN mkdir -p /app/logs
EXPOSE 80
ENV UVICORN_CMD="uvicorn src.api:app --host 0.0.0.0 --port 80 --reload"
CMD ["sh", "-c", "$UVICORN_CMD"]
