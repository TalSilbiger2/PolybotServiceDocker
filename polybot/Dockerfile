FROM python:3.10-slim
WORKDIR /usr/src/polybot
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip3 install -r requirements.txt
# When it should localhost:8081 and when it should be yolo5-service
ENV YOLO5_SERVICE_URL=http://yolo5-service:8081/predict
COPY . .
EXPOSE 8081
CMD ["python3", "app.py"]
