import boto3
import os
import uuid
from face_recognition import load_image_file, face_encodings
import pinecone

# Initialize Pinecone
pc = pinecone.Pinecone(
    api_key=os.environ.get("PINECONE_KEY")
)
index = pc.Index('usman-face-recognition')

# Initialize S3 client
s3 = boto3.client('s3')


def upload_image_and_update_database(bucket_name, image_path):
    # Generate a unique ID for the image
    image_uuid = str(uuid.uuid4())

    # Upload image to S3
    s3_key = f"images/{image_uuid}.jpg"
    try:
        s3.upload_file(image_path, bucket_name, s3_key)
        s3_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_key}"
        print(f"Uploaded {image_path} to {s3_url}")
    except Exception as e:
        print(f"Failed to upload {image_path} to S3: {e}")
        return

    # Generate face encoding
    try:
        image = load_image_file(image_path)
        encoding = face_encodings(image)
        if encoding:  # Check if face is detected
            vector = encoding[0].tolist()  # Convert numpy array to list
            # Save metadata to Pinecone
            index.upsert([(image_uuid, vector, {"s3_url": s3_url})])
            print(f"Metadata for {image_uuid} stored in Pinecone.")
        else:
            print(f"No face detected in {image_path}")
    except Exception as e:
        print(f"Failed to process {image_path}: {e}")

def process_directory(bucket_name, directory_path):
    for subdir, dirs, files in os.walk(directory_path):
        for file in files:
            filepath = os.path.join(subdir, file)
            if filepath.lower().endswith(('.png', '.jpg', '.jpeg')):  # Filter for image files
                try:
                    upload_image_and_update_database(bucket_name, filepath)
                except Exception as e:
                    print(f"Failed to process {filepath}: {str(e)}")

# Example usage
process_directory('usman-facial-recognition-images', '/Users/blake/Batch 1')