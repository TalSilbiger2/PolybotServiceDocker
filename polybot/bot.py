import telebot
from loguru import logger
import os
import time
from telebot.types import InputFile
import requests
import boto3
from pathlib import Path
from retrying import retry


class Bot:

    def __init__(self, token, telegram_chat_url):
        # create a new instance of the TeleBot class.
        # all communication with Telegram servers are done using self.telegram_bot_client
        self.telegram_bot_client = telebot.TeleBot(token)

        # remove any existing webhooks configured in Telegram servers
        self.telegram_bot_client.remove_webhook()
        time.sleep(0.5)

        # set the webhook URL
        self.telegram_bot_client.set_webhook(url=f'{telegram_chat_url}/{token}/', timeout=60)

        logger.info(f'Telegram Bot information\n\n{self.telegram_bot_client.get_me()}')

    def send_text(self, chat_id, text):
        self.telegram_bot_client.send_message(chat_id, text)

    def send_text_with_quote(self, chat_id, text, quoted_msg_id):
        self.telegram_bot_client.send_message(chat_id, text, reply_to_message_id=quoted_msg_id)

    def is_current_msg_photo(self, msg):
        return 'photo' in msg

    def download_user_photo(self, msg):
        """
        Downloads the photos that sent to the Bot to `photos` directory (should be existed)
        :return:
        """
        if not self.is_current_msg_photo(msg):
            raise RuntimeError(f'Message content of type \'photo\' expected')

        file_info = self.telegram_bot_client.get_file(msg['photo'][-1]['file_id'])
        data = self.telegram_bot_client.download_file(file_info.file_path)
        folder_name = file_info.file_path.split('/')[0]

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        with open(file_info.file_path, 'wb') as photo:
            photo.write(data)

        return file_info.file_path

    def send_photo(self, chat_id, img_path):
        if not os.path.exists(img_path):
            raise RuntimeError("Image path doesn't exist")

        self.telegram_bot_client.send_photo(
            chat_id,
            InputFile(img_path)
        )

    def handle_message(self, msg):
        """Bot Main message handler"""
        logger.info(f'Incoming message: {msg}')
        self.send_text(msg['chat']['id'], f'Your original message: {msg["text"]}')


class ObjectDetectionBot(Bot):


    def upload_to_s3(self, s3_client, msg, file_path, bucket_name, file_name):
        try:
            s3_client.upload_file(file_path, bucket_name, file_name)
            logger.info(f"The file was uploaded to S3, under the name: {file_name}")
        except Exception as e:
            logger.error(f"Failed to upload photo to S3: {e}")
            self.send_text(
                chat_id=msg['chat']['id'],
                text="Error: Could not upload the photo. Please try again later."
            )
            return


    def handle_message(self, msg):
        logger.info(f'Incoming message: {msg}')
        if self.is_current_msg_photo(msg):
            photo_path = self.download_user_photo(msg)

            # TODO upload the photo to S3
            file_path = str(photo_path)
            bucket_name = os.environ['BUCKET_NAME']
            file_name = f"photos/{photo_path.split('/')[-1]}"
            s3_client = boto3.client('s3')
            try:
                self.upload_to_s3(s3_client, msg, file_path, bucket_name, file_name)
            except Exception as e:
                logger.error(f"Error calling YOLO5 service: {e}")
                self.send_text(
                    chat_id=msg['chat']['id'],
                    text="Error: Could not upload the photo."
                )
                return

            # TODO send an HTTP request to the `yolo5` service for prediction
            image_url = f"https//{bucket_name}.s3.amazonaws.com/{file_name}"
            yolo5_service_url = "http://yolo5-service:8081/predict"

            try:
                # headers = {"Content-Type": "application/json"}
                response = requests.post(yolo5_service_url, params={"imgUrl": image_url})
                response.raise_for_status()
                prediction_results = response.json()
                logger.info(f"YOLO5 prediction results: {prediction_results}")
            except requests.exceptions.RequestException as e:
                logger.error(f"Error calling YOLO5 service: {e}")
                self.send_text(
                    chat_id=msg['chat']['id'],
                    text="Error: Could not process the photo. Please try again later."
                )
                return

            # TODO send the returned results to the Telegram end-user
            prediction_summary = self.format_prediction_summary(prediction_results)
            try:
                self.send_text(msg['chat']['id'], prediction_summary)
            except Exception as e:
                logger.error(f"Error sending message to Telegram user: {e}")
                self.send_text(
                    chat_id=msg['chat']['id'],
                    text="Error: Unable to send the prediction results. Please try again later."
                )


    def format_prediction_summary(self, prediction_results):
        """Formats the prediction results into a readable string for the user."""
        labels = prediction_results.get("labels", [])
        if not labels:
            return "No objects detected in the image."

        summary = "Detected objects:\n\n"
        for label in labels:
            summary += (
                f"Class: {label['class']}\n"
                f"Center: ({label['cx']}, {label['cy']})\n"
                f"Size: {label['width']} x {label['height']}\n\n"
            )
        return summary