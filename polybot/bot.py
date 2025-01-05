import telebot
from loguru import logger
import os
import time
from telebot.types import InputFile
import requests
import boto3
from pathlib import Path

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
    def handle_message(self, msg):
        """Handle the incoming messages, process the photos and interact with YOLO5 service"""
        logger.info(f'Incoming message: {msg}')

        if self.is_current_msg_photo(msg):
            # Message arrive from the user to the telegram bot, we first download it
            photo_path = self.download_user_photo(msg)

            bucket_name = os.environ['BUCKET_NAME']
            s3_client = boto3.client('s3')
            photo_key = f"uploaded_photos/{Path(photo_path).name}"

            try:
                # After download it, we will upload it to S3 bucket
                s3_client.upload_file(photo_path, bucket_name, photo_key)
                logger.info(f"Uploaded photo to S3: {bucket_name}/{photo_key}")
            except Exception as e:
                logger.error(f"Failed to upload photo to S3: {e}")
                self.send_text(
                    chat_id=msg['chat']['id'],
                    text="Error: Could not upload the photo. Please try again later."
                )
                return

            # Polybot service sends an HTTP request to the YOLO5 service for prediction
            yolo5_service_url = "http://yolo5-service:8081/predict"
            params = {"imgName": Path(photo_key).name}

            try:
                # Polybot sends the HTTP POST request
                response = requests.post(yolo5_service_url, params=params)
                # Polybot waits for response
                response.raise_for_status()
                # Polybot parse the JSON
                prediction_results = response.json()
                logger.info(f"YOLO5 prediction results: {prediction_results}")
            except Exception as e:
                logger.error(f"Error calling YOLO5 service: {e}")
                self.send_text(
                    chat_id=msg['chat']['id'],
                    text="Error: Could not process the photo. Please try again later."
                )
                return

            # Step 4: Send the returned results to the Telegram end-user
            prediction_summary = self.format_prediction_summary(prediction_results)
            try:
                # Polybot sends the prediction results
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


if __name__ == "__main__":
   # Load environment variables
   TELEGRAM_TOKEN = os.environ['TELEGRAM_TOKEN']
   TELEGRAM_APP_URL = os.environ['TELEGRAM_APP_URL']

   bot = ObjectDetectionBot(TELEGRAM_TOKEN, TELEGRAM_APP_URL)

   @bot.telegram_bot_client.message_handler(func=lambda message: True, content_types=['text', 'photo'])
   def on_message(message):
      bot.handle_message(message.json)

   logger.info("Starting the bot...")
   bot.telegram_bot_client.infinity_polling()