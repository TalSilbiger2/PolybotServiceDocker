import flask
from flask import request
import os
from bot import ObjectDetectionBot
import boto3

app = flask.Flask(__name__)


secrets_file_path = "/run/secrets/telegram_token"

with open(secrets_file_path, "r") as file:
    TELEGRAM_TOKEN =  file.read().strip()

TELEGRAM_APP_URL = os.environ['TELEGRAM_APP_URL']
S3_BUCKET_NAME = os.environ['BUCKET_NAME']
# Initialize the S3 client
s3_client = boto3.client('s3')
secret_file_path = '/run/secrets/telegram_token'


@app.route('/', methods=['GET'])
def index():
    return 'Ok'


@app.route(f'/{TELEGRAM_TOKEN}/', methods=['POST'])
def webhook():
    req = request.get_json()
    bot.handle_message(req['message'])
    return 'Ok'


if __name__ == "__main__":
    bot = ObjectDetectionBot(TELEGRAM_TOKEN, TELEGRAM_APP_URL, S3_BUCKET_NAME, s3_client)

    app.run(host='0.0.0.0', port=8443)
