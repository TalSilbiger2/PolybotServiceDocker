import time
from pathlib import Path
from flask import Flask, request, jsonify
from detect import run
import uuid
import yaml
from loguru import logger
import os
import boto3
from pymongo import MongoClient

# S3 client initialization
images_bucket = os.environ['BUCKET_NAME']
s3_client = boto3.client('s3')

# בדוק אם הצלחנו להתחבר ל-S3
try:
    response = s3_client.list_buckets()
    print("Buckets:", response['Buckets'])
except Exception as e:
    print("Error: couldnt connect to s3", e)

# MongoDB client initialization
mongo_uri = os.environ['MONGO_URI']  # MongoDB connection URI
mongo_client = MongoClient(mongo_uri)
db = mongo_client['predictions_db']
predictions_collection = db['predictions']

with open("data/coco128.yaml", "r") as stream:
    names = yaml.safe_load(stream)['names']

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    prediction_id = str(uuid.uuid4())
    logger.info(f'prediction: {prediction_id}. start processing')

    try:
        # Receives a URL parameter representing the image to download from S3
        img_url = request.args.get('imgName')
        logger.info(f"Received image name from request: {img_url}")

        img_name = request.args.get('imgName')
        logger.info(f"Received image name from request: {img_name}")

        # Extract the relative path from the full S3 URL
        img_name = img_url.split('.com/')[-1]  # This will get "photos/file_6.jpg"
        original_img_path = f"{img_name.split('/')[-1]}"  # This will get just "file_6.jpg"
        # TODO download img_name from S3, store the local image path in the original_img_path variable.
        #  The bucket name is provided as an env var BUCKET_NAME.

        s3_client.download_file(images_bucket, img_name, original_img_path)
        logger.info(f"Image {img_name} downloaded successfully to {original_img_path}")

    except Exception as e:
        logger.error(f"Failed to download image from bucket: {e}")
        return jsonify({"error": f"Failed to process image: {e}"}), 500

    logger.info(f'prediction: {prediction_id}/{original_img_path}. Download img completed')
    logger.info(f"Downloading image {img_name} from bucket {images_bucket} to {original_img_path}")

    # Predicts the objects in the image
    run(
        weights='yolov5s.pt',
        data='data/coco128.yaml',
        source=original_img_path,
        project='static/data',
        name=prediction_id,
        save_txt=True
    )
    logger.info(f"Running YOLOv5 on {original_img_path}")
    logger.info(f'prediction: {prediction_id}/{original_img_path}. done')
    logger.info(f"YOLOv5 model finished successfully on image {original_img_path}")



    # This is the path for the predicted image with labels
    # The predicted image typically includes bounding boxes drawn around the detected objects, along with class labels and possibly confidence scores.
    predicted_img_path = Path(f'static/data/{prediction_id}/{original_img_path}')
    logger.info(f"Predicted image saved at {predicted_img_path}")

    # TODO Uploads the predicted image (predicted_img_path) to S3 (be careful not to override the original image).
    # Upload predicted image to S3
    try:
        predicted_img_s3_path = f"predictions/{prediction_id}/{img_name}"
        s3_client.upload_file(str(predicted_img_path), images_bucket, predicted_img_s3_path)
        logger.info(f"Predicted image uploaded successfully to S3 at {predicted_img_s3_path}")
    except Exception as e:
        logger.error(f"Failed to upload predicted image: {e}")
        return f"Failed to upload predicted image: {e}", 500

    logger.info(f"Uploading predicted image to S3 at {predicted_img_s3_path}")

    # Parse prediction labels and create a summary
    pred_summary_path = Path(f'static/data/{prediction_id}/labels/{original_img_path.split(".")[0]}.txt')
    logger.info(f"Looking for prediction labels at {pred_summary_path}")

    logger.info(f"Checking if label file {pred_summary_path} exists after YOLOv5 run")
    if pred_summary_path.exists():
        logger.info(f"Label file {pred_summary_path} was successfully created.")
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
            'predicted_img_path': str(predicted_img_path),
            'labels': labels,
            'time': time.time()
        }

        # TODO store the prediction_summary in MongoDB
        try:
            result = predictions_collection.insert_one(prediction_summary)
            logger.info(f"Successfully stored prediction {prediction_id} in MongoDB")
        except Exception as e:
            logger.error(f"Failed to store prediction in MongoDB: {e}")
            return f"Failed to store prediction in MongoDB: {e}", 500

        prediction_summary['_id'] = str(result.inserted_id)

        try:
            os.remove(original_img_path)
            logger.info(f"Cleaned up temporary file: {original_img_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary file {original_img_path}: {e}")

        logger.info(f"Prediction {prediction_id} completed successfully")

        return jsonify({
            'prediction_id': prediction_id,
            'original_img_path': original_img_path,
            'predicted_img_path': str(predicted_img_path),
            'labels': labels,
            'time': time.time(),
            '_id': str(result.inserted_id)  # המרת ה-ID ב-MongoDB לפורמט string
        })

    else:
        logger.warning(f"Failed to find prediction {prediction_id}/{original_img_path}")
        #return f'prediction: {prediction_id}/{original_img_path}. prediction result not found', 404
        #send the right error to the user
        error_message = (
            f"Error - the model could not find any objects in your image."
            f"\nPrediction ID: {prediction_id}"
            f"\nImage: {original_img_path}"
        )
        logger.warning(error_message)
        return error_message, 404

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8081)