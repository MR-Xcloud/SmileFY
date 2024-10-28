import requests
from django.conf import settings
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework.views import APIView
from rest_framework import status
import cv2
import numpy as np
import face_alignment
from skimage.exposure import match_histograms
# from google.colab.patches import cv2_imshow
from PIL import Image, ImageEnhance
import os


class imageapi(APIView):

    def post(self, request, *args, **kwargs):
        # Check for file1 and file2 in request files
        file1 = request.FILES.get('file1')
        print(file1)
        file2 = request.FILES.get('file2')  # This can either be a file or we'll expect a URL

        # Log the request data for debugging
        print(f"Request Data: {request.data}")

        # Check if file1 is provided
        if not file1:
            return Response({"error": "File1 must be provided."}, status=status.HTTP_400_BAD_REQUEST)

        # Handle file2 as a URL or file
        file2_array = None
        if not file2:
            # Check if file2 is provided as a URL in the data
            file2_url = request.data.get('file2')  # Expecting 'file2_url' in the request body
            if not file2_url:
                return Response({"error": "File2 (either a file or a URL) must be provided."}, status=status.HTTP_400_BAD_REQUEST)

            try:
                # Download the image from the provided URL
                response = requests.get(file2_url)
                if response.status_code == 200:
                    file2_array = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
                else:
                    return Response({"error": "Could not retrieve the image from the URL."}, status=status.HTTP_400_BAD_REQUEST)
            except Exception as e:
                return Response({"error": f"Error downloading image from URL: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)
        else:
            # Convert the uploaded file2 to OpenCV array
            try:
                file2_array = cv2.imdecode(np.frombuffer(file2.read(), np.uint8), cv2.IMREAD_COLOR)
            except Exception as e:
                return Response({"error": f"Error reading file2: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)

        # Convert file1 to OpenCV array
        try:
            file1_array = cv2.imdecode(np.frombuffer(file1.read(), np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            return Response({"error": f"Error reading file1: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)

        if file1_array is None or file2_array is None:
            return Response({"error": "Could not decode one or both images."}, status=status.HTTP_400_BAD_REQUEST)

        # Save the files
        save_path1 = r"smily\images\file1.jpg"
        save_path2 = r"smily\images\file2.jpg"
        cv2.imwrite(save_path1, file1_array)
        cv2.imwrite(save_path2, file2_array)
        print("File1 and File2 saved successfully")

        # Initialize face alignment model
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device='cpu')

        # Load the input image (file1)
        img = cv2.imread(save_path1)
        if img is None:
            return Response({'error': 'Could not load the input image.'}, status=status.HTTP_400_BAD_REQUEST)

        # Resize the input image
        img = cv2.resize(img, (316, 421))

        # Convert the resized image to RGB
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect landmarks
        landmarks = fa.get_landmarks(rgb_img)
        if landmarks is None:
            return Response({'error': 'No landmarks detected.'}, status=status.HTTP_400_BAD_REQUEST)

        # Load the replacement image (file2)
        replacement_img = cv2.imread(save_path2)
        if replacement_img is None:
            return Response({'error': 'Could not load the replacement image.'}, status=status.HTTP_400_BAD_REQUEST)

        # Function to detect if the face in file1 is smiling
        def is_smiling(landmark_set):
            mouth_top = np.mean([landmark_set[i][1] for i in range(50, 53)])
            mouth_bottom = np.mean([landmark_set[i][1] for i in range(65, 68)])
            mouth_left = landmark_set[48][0]
            mouth_right = landmark_set[54][0]

            # Check if the distance between top and bottom lips suggests a smile
            if (mouth_bottom - mouth_top) > 5 and (mouth_right - mouth_left) > 40:
                return True
            return False

        # Process landmarks and check if the face is smiling
        is_smile_detected = False
        for landmark_set in landmarks:
            if is_smiling(landmark_set):
                is_smile_detected = True
                break

        if not is_smile_detected:
            return Response({'error': 'The face in file1 is not smiling. Please upload a smiling face image.'}, status=status.HTTP_400_BAD_REQUEST)

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
                print("Error: Invalid bounding box dimensions for the lip region.")
                return original_img

            replacement_img_resized = cv2.resize(replacement_img, (w, h))
            replacement_img_resized = adjust_teeth_color(original_img, replacement_img_resized, lip_points)
            mask = 255 * np.ones(replacement_img_resized.shape, replacement_img_resized.dtype)
            center = (x + w // 2, y + h // 2)

            if replacement_img_resized.dtype != np.uint8:
                replacement_img_resized = (replacement_img_resized * 255).astype(np.uint8)

            blended = cv2.seamlessClone(replacement_img_resized, original_img, mask, center, cv2.NORMAL_CLONE)
            return blended

        # Process landmarks and replace teeth
        for landmark_set in landmarks:
            lip_points = np.array([[int(landmark_set[i][0]), int(landmark_set[i][1])] for i in range(48, 68)], dtype=np.int32)
            if lip_points.size == 0:
                print("Error: No valid lip points detected.")
                continue

            img = replace_teeth_seamless(img, replacement_img, lip_points)

        # Adjust contrast and brightness
        final_img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        final_img_pil = Image.fromarray(final_img_rgb)
        contrast_enhancer = ImageEnhance.Contrast(final_img_pil)
        final_img_pil = contrast_enhancer.enhance(1)
        brightness_enhancer = ImageEnhance.Brightness(final_img_pil)
        final_img_pil = brightness_enhancer.enhance(1)

        # Convert back to OpenCV format
        final_img = cv2.cvtColor(np.array(final_img_pil), cv2.COLOR_RGB2BGR)

        # Save the final image
        save_path_final = r"static\images\final.jpg"
        cv2.imwrite(save_path_final, final_img)
        print("Final image saved successfully.")

        # Return the final image's URL
        final_image_url = os.path.join(settings.STATIC_URL, 'images/final.jpg')
        return Response({'final_image_url': final_image_url}, status=status.HTTP_200_OK)






