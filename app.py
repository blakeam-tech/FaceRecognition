import os
import uuid
import boto3
import pinecone
import gradio as gr
from face_recognition import load_image_file, face_encodings
from PIL import Image
import io
from datetime import datetime

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=os.environ.get("PINECONE_KEY"))
index = pc.Index(os.environ.get("PINECONE_INDEX"))

# Initialize S3 client
s3 = boto3.client('s3')

def upload_image_to_s3(bucket_name, image_path):
    s3_key = f"images/{image_path}"
    s3.upload_file(image_path, bucket_name, s3_key)
    return s3_key

def get_image_from_s3(bucket_name, s3_url):
    key = s3_url.split('/')[-1] if s3_url.startswith('http') else s3_url
    response = s3.get_object(Bucket=bucket_name, Key=f"images/{key}")
    return Image.open(response['Body'])

def delete_image_from_s3(bucket_name, image_path):
    s3_key = f"images/{image_path}.jpg"
    s3.delete_object(Bucket=bucket_name, Key=s3_key)

def handle_image_database_and_s3(bucket_name, image, old_image_uuid=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_uuid = old_image_uuid if old_image_uuid else str(uuid.uuid4())
    image_path = f"{image_uuid}_{timestamp}.jpg"
    image.save(image_path)
    
    new_s3_key = upload_image_to_s3(bucket_name, image_path)
    s3_url = f"https://{bucket_name}.s3.amazonaws.com/{new_s3_key}"
    
    if new_s3_key:
        encoding = face_encodings(load_image_file(image_path))[0].tolist()
        if old_image_uuid:
            current_entry = index.fetch(ids=[image_uuid]).get('vectors', {}).get(image_uuid, {})
            current_images = current_entry.get('metadata', {}).get('s3_urls', [])
            current_images.append(s3_url)
            index.upsert([(image_uuid, encoding, {"s3_urls": current_images})])
        else:
            index.upsert([(image_uuid, encoding, {"s3_urls": [s3_url]})])
        os.remove(image_path)
        return "Image processed successfully.", s3_url, image_uuid
    else:
        os.remove(image_path)
        return "Failed to upload image.", None, None

def search_for_similar_image(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    encoding = face_encodings(load_image_file(io.BytesIO(img_byte_arr.getvalue())))
    if encoding:
        query_vector = encoding[0].tolist()
        query_result = index.query(vector=query_vector, top_k=1, include_metadata=True)
        if query_result['matches'] and query_result['matches'][0]['score'] > 0.95:
            best_match = query_result['matches'][0]
            image_uuid = best_match['id']
            metadata = index.fetch(ids=[image_uuid]).get('vectors', {}).get(image_uuid, {}).get('metadata', {})
            s3_urls = metadata.get('s3_urls', [])
            if not s3_urls:
                s3_urls = metadata.get('s3_url', [])
                if s3_urls:
                    s3_urls = [s3_urls]
            return True, f"Match found. File ID: {image_uuid}. Do you want to update this child's profile or create a new ID?", s3_urls, image_uuid
        else:
            return False, "No close matches found. Create a new ID?", [], None
    else:
        return False, "No face detected in the image.", [], None

def handle_update(image, old_uuid):
    """
    Handles the update of an image in the database and returns a formatted message
    if the update is successful.
    """
    message, s3_url, image_uuid = handle_image_database_and_s3(os.environ.get('AWS_BUCKET'), image, old_uuid)
    if message.startswith("Updated"):
        return f"Updated image in database. Updated File ID: {image_uuid}."
    return message  # Returns the error message if update is not successful.

def search_and_display(image):
    found, message, s3_urls, old_uuid = search_for_similar_image(image)
    if found:
        return message, s3_urls, old_uuid
    else:
        # Handle cases where no match is found or a new UUID needs to be created
        return message, [], ""
    
with gr.Blocks() as app:
    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Image")
        submit_button = gr.Button("Search for Similar Image")
        
    result_text = gr.Text(label="Result")
    images_gallery = gr.Gallery(label="Images of Child")
    old_uuid_text = gr.Textbox(visible=False, label="UUID of Matched Image")
    
    with gr.Row():
        add_button = gr.Button("Add to Existing Child")
        new_button = gr.Button("Create New ID")
        
    def search_and_display(image):
        found, message, s3_urls, old_uuid = search_for_similar_image(image)
        return message, s3_urls, old_uuid

    def handle_addition(image, old_uuid):
        message, s3_url, _ = handle_image_database_and_s3(os.environ.get("AWS_BUCKET"), image, old_uuid)
        return f"Image added to child's file. File ID: {old_uuid}."

    def handle_new(image):
        message, s3_url, new_uuid = handle_image_database_and_s3(os.environ.get("AWS_BUCKET"), image)
        return f"New record created for child. File ID: {new_uuid}."

    submit_button.click(
        fn=search_and_display,
        inputs=image_input,
        outputs=[result_text, images_gallery, old_uuid_text]
    )
    
    add_button.click(
        fn=handle_addition,
        inputs=[image_input, old_uuid_text],
        outputs=result_text
    )
    
    new_button.click(
        fn=handle_new,
        inputs=image_input,
        outputs=result_text
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=int(os.environ.get('PORT', 7860)), debug=False)