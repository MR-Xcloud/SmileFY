from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import cv2
import numpy as np
import requests
import face_alignment
from django.conf import settings
from skimage.exposure import match_histograms
import boto3
import uuid
from rest_framework.exceptions import ValidationError
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import os

class imageapi(APIView):
    def post(self, request):
        # Check for file1 and file2 in request files
        file1 = request.FILES.get('file1')
        print("file1 saved")
        file2 = request.FILES.get('file2')  # This can either be a file or a URL

        if not file1:
            return Response({"error": "File1 must be provided."}, status=status.HTTP_400_BAD_REQUEST)

        # Upload file1 to S3 first and get its URL
        try:
            file1_url = self.upload_to_s3(file1, is_file1=True)  # Upload file1 to S3
        except Exception as e:
            return Response({"error": f"Failed to upload file1 to S3: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Save file1 to a specified location before reading it into OpenCV
        file1_save_path = r"static/images/file1.jpg"
        with open(file1_save_path, 'wb') as f:
            for chunk in file1.chunks():
                f.write(chunk)

        # Convert file1 to OpenCV array
        try:
            file1_array = cv2.imread(file1_save_path)
        except Exception as e:
            return Response({"error": f"Error reading file1: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)

        # Handle file2 as a URL or file
        file2_array = None
        if not file2:
            file2_url = request.data.get('file2')  # Check for file2 URL if not uploaded as file
            if not file2_url:
                return Response({"error": "File2 (either a file or a URL) must be provided."}, status=status.HTTP_400_BAD_REQUEST)

            try:
                response = requests.get(file2_url)
                if response.status_code == 200:
                    file2_array = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
                else:
                    return Response({"error": "Could not retrieve the image from the URL."}, status=status.HTTP_400_BAD_REQUEST)
            except Exception as e:
                return Response({"error": f"Error downloading image from URL: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)
        else:
            try:
                file2_array = cv2.imdecode(np.frombuffer(file2.read(), np.uint8), cv2.IMREAD_COLOR)
            except Exception as e:
                return Response({"error": f"Error reading file2: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)

        if file1_array is None or file2_array is None:
            return Response({"error": "Could not decode one or both images."}, status=status.HTTP_400_BAD_REQUEST)

        # Save the original size of file1
        original_size = file1_array.shape[:2]  # (height, width)

        # Resize file1 to (316, 421) for teeth replacement processing
        resized_file1 = cv2.resize(file1_array, (316, 421))

        # Initialize face alignment model
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device='cpu')

        # Convert the resized image to RGB
        rgb_img = cv2.cvtColor(resized_file1, cv2.COLOR_BGR2RGB)

        # Detect landmarks
        landmarks = fa.get_landmarks(rgb_img)
        if landmarks is None:
            return Response({'error': 'No landmarks detected.'}, status=status.HTTP_400_BAD_REQUEST)

        # Function to adjust teeth color to match surrounding lips
        def adjust_teeth_color(original_img, replacement_img, lip_points):
            (x, y, w, h) = cv2.boundingRect(lip_points)
            original_crop = original_img[y:y+h, x:x+w]
            replacement_img_adj = match_histograms(replacement_img, original_crop, channel_axis=-1)
            return replacement_img_adj

        # Function for seamless cloning
        def replace_teeth_seamless(original_img, replacement_img, lip_points):
            (x, y, w, h) = cv2.boundingRect(lip_points)
            x = max(0, x)
            y = max(0, y)
            w = min(w, original_img.shape[1] - x)
            h = min(h, original_img.shape[0] - y)

            if w == 0 or h == 0:
                return original_img

            replacement_img_resized = cv2.resize(replacement_img, (w, h))
            replacement_img_resized = adjust_teeth_color(original_img, replacement_img_resized, lip_points)
            mask = 255 * np.ones(replacement_img_resized.shape, replacement_img_resized.dtype)
            center = (x + w // 2, y + h // 2)

            blended = cv2.seamlessClone(replacement_img_resized, original_img, mask, center, cv2.NORMAL_CLONE)
            return blended

        # Process landmarks and replace teeth
        replacement_img = file2_array  # Use file2_array directly (as it's already read into an image)
        for landmark_set in landmarks:
            lip_points = np.array([[int(landmark_set[i][0]), int(landmark_set[i][1])] for i in range(48, 68)], dtype=np.int32)
            if lip_points.size == 0:
                continue

            resized_file1 = replace_teeth_seamless(resized_file1, replacement_img, lip_points)

        # After teeth replacement, resize the processed image back to the original size
        final_img = cv2.resize(resized_file1, (original_size[1], original_size[0]))  # Resize to (width, height)

        # Upload the final image to S3 and return its URL
        try:
            final_image_url = self.upload_to_s3(final_img)  # Upload the final image to S3

            return Response({
                'before_image_url': file1_url,  # Original file1 image URL
                'after_image_url': final_image_url  # Final processed image URL
            }, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": f"Failed to upload images to S3: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def upload_to_s3(self, file, is_file1=False):
        """Helper method to upload images to S3 and return the URL."""
        try:
            s3_client = boto3.client('s3',
                                     aws_access_key_id=settings.AWS_ACCESS_KEY,
                                     aws_secret_access_key=settings.AWS_SECRET_KEY,
                                     region_name=settings.AWS_REGION)

            # Check if input is a file or NumPy array
            if isinstance(file, np.ndarray):  # It's a NumPy array
                file_name = f"{str(uuid.uuid4())}.jpg"  # Generate a random file name
                _, buffer = cv2.imencode('.jpg', file)
                file_data = buffer.tobytes()

                content_type = 'image/jpeg'  # For NumPy array, default to jpeg
            else:  # It's a file from request.FILES
                file_name = file.name  # Original file name
                file_data = file.read()

                # Dynamically set the content type based on file extension
                if file_name.lower().endswith(('.jpg', '.jpeg')):
                    content_type = 'image/jpeg'
                elif file_name.lower().endswith('.png'):
                    content_type = 'image/png'
                else:
                    raise ValidationError("Unsupported file type.")

            # Upload the image to S3 with ContentType for direct viewing in browser
            bucket_name = 'fetch-delivery'  # Set your S3 bucket name
            s3_client.put_object(Bucket=bucket_name, Key=file_name, Body=file_data, ContentType=content_type)

            # Generate the file URL
            file_url = f"https://{bucket_name}.s3.amazonaws.com/{file_name}"

            return file_url

        except Exception as e:
            raise Exception(f"Error uploading to S3: {str(e)}")
