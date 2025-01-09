import telebot
from loguru import logger
import os
import time
from telebot.types import InputFile
import requests
import boto3
from botocore.exceptions import NoCredentialsError

s3_client = boto3.client('s3')

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
        try:
            file_info = self.telegram_bot_client.get_file(msg['photo'][-1]['file_id'])
            data = self.telegram_bot_client.download_file(file_info.file_path)
            folder_name = file_info.file_path.split('/')[0]

            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            with open(file_info.file_path, 'wb') as photo:
                photo.write(data)

            return file_info.file_path

        except Exception as e:
            logger.error(f"Failed to download image: {e}")
            raise RuntimeError(f"Failed to download image: {e}")

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

def upload_to_s3(file_path, bucket_name, s3_file_name):
    try:
        s3_client.upload_file(file_path, bucket_name, s3_file_name)
        logger.info(f"File uploaded to S3: {s3_file_name}")
        return True
    except NoCredentialsError:
        logger.error("Credentials not available")
        return False
    except Exception as e:
        logger.error(f"Failed to upload file to S3: {e}")
        raise RuntimeError(f"Failed to upload file to S3: {e}")


# שליחת בקשה ל-YOLOv5
def get_prediction_from_yolo5(image_url):
    yolo_port=os.environ['YOLO_PORT']
    url = f"http://yolo5-service:{yolo_port}/predict"
    params = {'imgName': image_url}  # שלח את כתובת התמונה ששמורה ב-S3

    try:
        response = requests.post(url, params=params)
        if response.status_code == 200:
            return response.json()  # יחזיר את התוצאה מ-YOLOv5
        else:
            logger.error(f"Error: {response.status_code}")
            raise RuntimeError(f"Error: {response.status_code} - {response.text}")

    except Exception as e:
        logger.error(f"Failed to get prediction from YOLOv5: {e}")
        raise RuntimeError(f"Failed to get prediction from YOLOv5: {e}")


class ObjectDetectionBot(Bot):
    def handle_message(self, msg):
        logger.info(f'Incoming message: {msg}')

        if self.is_current_msg_photo(msg):
            # שלב 1: הורדת התמונה מהמשתמש
            photo_path = self.download_user_photo(msg)
            logger.info(f'photo path is: {photo_path}')


            # TODO upload the photo to S3
            # שלב 2: העלאת התמונה ל-S3
            photo_path = str(photo_path)
            s3_bucket = os.environ['BUCKET_NAME']
            s3_file_name = f"photos/{photo_path.split('/')[-1]}"
            upload_to_s3(photo_path, s3_bucket, s3_file_name)

            # TODO send an HTTP request to the `yolo5` service for prediction
            s3_file_name = str(s3_file_name)
            image_url = f"https://{s3_bucket}.s3.amazonaws.com/{s3_file_name}"
            image_url = str(image_url)
            logger.info(f's3_file_name is: {s3_file_name} variable type: {type(s3_file_name)}')
            try:
                prediction_result = get_prediction_from_yolo5(image_url)

            # TODO send the returned results to the Telegram end-user
                if not prediction_result or 'labels' not in prediction_result or len(prediction_result['labels']) == 0:
                    error_message = "Sorry, no objects were detected in the image."
                    logger.info(f"No objects detected, sending message: {error_message}")  # לוג
                    self.send_text(msg['chat']['id'], error_message)
                    return  # חזרה מבלי לנסות שוב
                    # שלב 5: שליחת התוצאות למשתמש
                logger.info(f"Prediction results: {prediction_result}")  # לוג נוסף
                self.send_prediction_result(msg['chat']['id'], prediction_result)
            except Exception as e:
                error_message = "Sorry, no objects were detected in the image"
                logger.error(error_message)
                self.send_text(msg['chat']['id'], error_message)  # שלח למשתמש את הודעת השגיאה
    def send_prediction_result(self, chat_id, prediction_result):
        """
        שולח את תוצאות הזיהוי כטקסט למשתמש ב-Telegram
        """
        object_count = {}
        for label in prediction_result['labels']:
            object_name = label['class']
            object_count[object_name] = object_count.get(object_name, 0) + 1

        message = "Detected objects:\n"
        for obj, count in object_count.items():
            message += f"{obj}: {count}\n"

        self.send_text(chat_id, message)

    def send_prediction_image(self, chat_id, image_path):
        """
        שולח את התמונה עם התוצאות למשתמש ב-Telegram
        """
        self.send_photo(chat_id, image_path)