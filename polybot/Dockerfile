FROM python:3.10-slim
WORKDIR /usr/src/polybot
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip3 install -r requirements.txt
COPY . .
EXPOSE 8081
CMD ["python3", "app.py"]
