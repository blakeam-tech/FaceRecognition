# app.py
"""
This module sets up a Gradio interface for a face recognition app. 
It allows users to upload images and search for similar images using the functionality provided by face_recognition_core.
"""

import os
import gradio as gr
from face_recognition_core import search_and_display

with gr.Blocks() as app:
    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Image")
        submit_button = gr.Button("Search for Similar Image")

    result_text = gr.Text(label="Result")
    images_gallery = gr.Gallery(label="Images of Child")

    submit_button.click(
        fn=search_and_display, inputs=image_input, outputs=[result_text, images_gallery]
    )

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)), debug=False
    )
