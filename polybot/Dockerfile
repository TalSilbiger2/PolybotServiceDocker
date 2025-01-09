FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt /app/
# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Copy application files
COPY . /app/
# Expose port
EXPOSE 8081
# Run the application
CMD ["python", "app.py"]