# EchoSearch

EchoSearch is a powerful image recognition system designed to assist a charity organization in managing and identifying photos of missing children. This system leverages advanced facial recognition technology to locate matches effectively, enhancing the capabilities of search and rescue operations.

## Features

- **Facial Recognition**: Uses a Python-based facial recognition library to generate face encodings and a Pinecone database for retrieval.
- **Image Storage**: Integrates with AWS S3 for secure image storage.
- **User Interface**: Features a user-friendly interface built with Streamlit.
- **Deployment**: Deployed on Heroku for robust cloud hosting.

## Technology Stack

- **Python**: Primary programming language used for developing the core facial recognition functionality.
- **Streamlit**: Empowers the development of the frontend interface to interact with the system seamlessly.
- **Pinecone**: Vector database for image retrieval based on cosine similarity with face encoding.
- **AWS S3**: Used for storing and retrieving images securely.
- **Heroku**: Hosts the application.

## Local Development

To set up **EchoSearch** for local development, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourgithub/echosearch.git
   cd echosearch
   ```
2. **Install Dependencies**:
```bash
   pip install -r requirements.txt
   ```
3. **Environment Variables**:
Create a **.env** file in the root directory and add the following:
```bash
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key
PINECONE_API_KEY=your_pinecone_api_key
```
4. **Run the Application**:
```bash
streamlit run app_interface.py
```

### We welcome contributions to EchoSearch! If you have suggestions or improvements, please fork the repository and submit a pull request.

## License

EchoSearch is open-source software licensed under MIT.