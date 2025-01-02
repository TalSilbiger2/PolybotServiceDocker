import time
from pathlib import Path

import boto3
from flask import Flask, request
from detect import run
from pymongo import MongoClient
import uuid
import yaml
from loguru import logger
import os


images_bucket = os.environ['BUCKET_NAME']

with open("data/coco128.yaml", "r") as stream:
    names = yaml.safe_load(stream)['names']

app = Flask(__name__)

# Initialize S3 client
s3 = boto3.client('s3')

# Initialize MongoDB client
mongo_client = MongoClient(os.getenv('MONGO_URI'))
db = mongo_client.predictions  # Replace with your DB name
predictions_collection = db.predictions


@app.route('/predict', methods=['POST'])
def predict():
    # Generates a UUID for this current prediction HTTP request. This id can be used as a reference in logs to identify and track individual prediction requests.
    prediction_id = str(uuid.uuid4())

    logger.info(f'prediction: {prediction_id}. start processing')

    # Receives a URL parameter representing the image to download from S3
    img_name = request.args.get('imgName')

    bucket_name = os.getenv('BUCKET_NAME')
    # TODO download img_name from S3, store the local image path in the original_img_path variable.
    #  The bucket name is provided as an env var BUCKET_NAME.
    original_img_path = img_name
    try:
        s3.download_file(bucket_name, img_name, original_img_path)
        logger.info(f'prediction: {prediction_id}/{original_img_path}. Download img completed')
    except Exception as e:
        logger.error(f'Error downloading image from S3: {e}')
        return f'Error downloading image: {e}', 500

    logger.info(f'prediction: {prediction_id}/{original_img_path}. Download img completed')

    # Predicts the objects in the image
    run(
        weights='yolov5s.pt',
        data='data/coco128.yaml',
        source=original_img_path,
        project='static/data',
        name=prediction_id,
        save_txt=True
    )

    logger.info(f'prediction: {prediction_id}/{original_img_path}. done')

    # This is the path for the predicted image with labels
    # The predicted image typically includes bounding boxes drawn around the detected objects, along with class labels and possibly confidence scores.
    predicted_img_path = Path(f'static/data/{prediction_id}/{original_img_path}')

    # TODO Uploads the predicted image (predicted_img_path) to S3 (be careful not to override the original image).
    predicted_img_path = Path(f'static/data/{prediction_id}/labels/{img_name}')

    predicted_s3_key = f'predictions/{prediction_id}/{img_name}'
    try:
        s3.upload_file(str(predicted_img_path), bucket_name, predicted_s3_key)
        logger.info(f'prediction: {prediction_id}/{predicted_img_path}. Upload completed')
    except Exception as e:
        logger.error(f'Error uploading predicted image to S3: {e}')
        return f'Error uploading predicted image: {e}', 500


    # Parse prediction labels and create a summary
    pred_summary_path = Path(f'static/data/{prediction_id}/labels/{original_img_path.split(".")[0]}.txt')
    if pred_summary_path.exists():
        with open(pred_summary_path) as f:
            labels = f.read().splitlines()
            labels = [line.split(' ') for line in labels]
            labels = [{
                'class': names[int(l[0])],
                'cx': float(l[1]),
                'cy': float(l[2]),
                'width': float(l[3]),
                'height': float(l[4]),
            } for l in labels]

        logger.info(f'prediction: {prediction_id}/{original_img_path}. prediction summary:\n\n{labels}')

        prediction_summary = {
            'prediction_id': prediction_id,
            'original_img_path': original_img_path,
            'predicted_img_path': predicted_img_path,
            'labels': labels,
            'time': time.time()
        }

        # TODO store the prediction_summary in MongoDB
        try:
            predictions_collection.insert_one(prediction_summary)
            logger.info(f'prediction: {prediction_id}. Summary stored in MongoDB')
        except Exception as e:
            logger.error(f'Error storing prediction summary in MongoDB: {e}')
            return f'Error storing prediction summary: {e}', 500


        return prediction_summary
    else:
        return f'prediction: {prediction_id}/{original_img_path}. prediction result not found', 404


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8081)
