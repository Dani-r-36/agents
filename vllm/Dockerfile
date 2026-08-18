# 1. Use a clean, lightweight Python base image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy application files into the container
COPY . .

# 4. Upgrade pip and install dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# 5. Expose the port FastAPI will run on
EXPOSE 8080

# 6. Start the server (replace 'server:app' with 'filename:app' if your file isn't named server.py)
CMD ["uvicorn", "llm_docker:app", "--host", "0.0.0.0", "--port", "8080"]