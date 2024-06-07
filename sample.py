import pinecone
import os
from face_recognition import load_image_file, face_encodings

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=os.environ.get("PINECONE_KEY"))
index = pc.Index('usman-face-recognition')

def search_for_similar_images(image_path):
    image = load_image_file(image_path)
    encodings = face_encodings(image)
    if encodings:
        query_vector = encodings[0].tolist()
        print(query_vector)
        query_result = index.query(vector=query_vector, top_k=1, include_metadata=True)
        print(query_result)
        """
        if query_result['matches']:
            match = query_result['matches'][0]
            return f"Match found: ID: {match['id']}, Score: {match['score']}"
        else:
            return "No similar face found."
        """
    else:
        return "No face detected in the image."
search_for_similar_images('/Users/blake/Downloads/a958ad72-401b-4742-b68b-2714f0434f67.jpg')