# face_recognition_core.py
import io
import os
import uuid
from datetime import datetime

from PIL import Image
from face_recognition import load_image_file, face_encodings
import boto3
import pinecone

# Initialize Pinecone and S3 client
pc = pinecone.Pinecone(api_key=os.environ.get("PINECONE_KEY"))
index = pc.Index(os.environ.get("PINECONE_INDEX"))
s3 = boto3.client("s3")


def upload_image_to_s3(bucket_name, image_path):
    """Uploads an image to an S3 bucket and returns the S3 key."""
    s3_key = f"images/{image_path}"
    s3.upload_file(image_path, bucket_name, s3_key)
    return s3_key


def get_image_from_s3(bucket_name, s3_url):
    """Retrieves an image from an S3 bucket given a URL."""
    key = s3_url.split("/")[-1] if s3_url.startswith("http") else s3_url
    response = s3.get_object(Bucket=bucket_name, Key=f"images/{key}")
    return Image.open(response["Body"])


def handle_image_database_and_s3(bucket_name, image, old_image_uuid=None):
    """Handles uploading an image to S3 and updating the database."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_uuid = old_image_uuid if old_image_uuid else str(uuid.uuid4())
    image_path = f"{image_uuid}_{timestamp}.jpg"
    image.save(image_path)
    new_s3_key = upload_image_to_s3(bucket_name, image_path)
    s3_url = f"https://{bucket_name}.s3.amazonaws.com/{new_s3_key}"
    if new_s3_key:
        encoding = face_encodings(load_image_file(image_path))[0].tolist()
        index.upsert(
            [
                (
                    image_uuid,
                    encoding,
                    {"s3_urls": [s3_url]}
                    if old_image_uuid is None
                    else {"s3_urls": [s3_url]},
                )
            ]
        )
        os.remove(image_path)
        return "Image processed successfully.", s3_url, image_uuid
    os.remove(image_path)
    return "Failed to upload image.", None, None


def search_for_similar_image(image):
    """Searches for a similar image using face encodings."""
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="JPEG")
    encoding = face_encodings(load_image_file(io.BytesIO(img_byte_arr.getvalue())))
    if encoding:
        query_vector = encoding[0].tolist()
        query_result = index.query(vector=query_vector, top_k=1, include_metadata=True)
        if query_result["matches"]:
            best_match = query_result["matches"][0]
            image_uuid = best_match["id"]
            metadata = (
                index.fetch(ids=[image_uuid])
                .get("vectors", {})
                .get(image_uuid, {})
                .get("metadata", {})
            )
            s3_urls = metadata.get("s3_urls", [])
            return True, f"Match found. File ID: {image_uuid}", s3_urls, image_uuid
    return False, "No face detected or no close matches found.", [], None
