FROM python:3.10-slim
WORKDIR /usr/src/polybot
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip3 install -r requirements.txt
# Install AWS CLI
RUN apt-get update && apt-get install -y \
    unzip \
    curl \
    && curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -rf awscliv2.zip
COPY . .
EXPOSE 8081
EXPOSE 8443

ENV TELEGRAM_TOKEN=${TELEGRAM_TOKEN}
ENV TELEGRAM_APP_URL=${TELEGRAM_APP_URL}
ENV BUCKET_NAME=${BUCKET_NAME}
ENV AWS_ACCESS_KEY=${AWS_ACCESS_KEY}
ENV AWS_SECRET_KEY=${AWS_SECRET_KEY}

CMD ["python3", "app.py"]