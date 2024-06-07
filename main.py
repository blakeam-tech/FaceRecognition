from pinecone import Pinecone, ServerlessSpec
import os

pc = Pinecone(api_key=os.getenv('PINECONE_KEY'))

pc.create_index(
    name="usman-face-recognition",
    dimension=128, # Replace with your model dimensions
    metric="cosine", # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)