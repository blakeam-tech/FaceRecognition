# test_s3_operations.py
import os
from unittest.mock import patch, MagicMock
from face_recognition_core import upload_image_to_s3, get_image_from_s3


def test_upload_image_to_s3():
    """Test that an image is uploaded to S3 correctly."""
    with patch("boto3.client") as mock_boto3:
        mock_s3 = MagicMock()
        mock_s3.upload_file.return_value = None
        mock_boto3.return_value = mock_s3

        bucket_name = os.getenv("AWS_BUCKET_NAME")
        image_path = "test/image.jpg"
        expected_s3_key = f"images/{image_path}"

        s3_key = upload_image_to_s3(bucket_name, image_path)

        assert s3_key == expected_s3_key
        mock_s3.upload_file.assert_called_once_with(
            image_path, bucket_name, expected_s3_key
        )


def test_get_image_from_s3():
    """Test retrieving an image from S3 using a URL."""
    with patch("boto3.client") as mock_boto3:
        mock_s3 = MagicMock()
        mock_response = {"Body": MagicMock()}
        mock_response["Body"].read.return_value = b"test image data"
        mock_s3.get_object.return_value = mock_response
        mock_boto3.return_value = mock_s3

        bucket_name = os.getenv("AWS_BUCKET_NAME")
        s3_url = "http://example.com/test/image.jpg"
        image = get_image_from_s3(bucket_name, s3_url)

        assert image is not None
