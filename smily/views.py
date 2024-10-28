from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import cv2
import numpy as np
import os
import requests
import face_alignment
from django.conf import settings
from skimage.exposure import match_histograms

class imageapi(APIView):
    def post(self, request):
        # Check for file1 and file2 in request files
        file1 = request.FILES.get('file1')
        print("=============Start")
        file2 = request.FILES.get('file2')  # This can either be a file or we'll expect a URL

        if not file1:
            return Response({"error": "File1 must be provided."}, status=status.HTTP_400_BAD_REQUEST)

        # Save file1 to a specified location before reading it into OpenCV
        file1_save_path = r"static\images\file1.jpg"
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
            file2_url = request.data.get('file2')
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
        replacement_img = cv2.imread(r"smily\images\file2.jpg")  # Ensure save path for file2 is correct
        for landmark_set in landmarks:
            lip_points = np.array([[int(landmark_set[i][0]), int(landmark_set[i][1])] for i in range(48, 68)], dtype=np.int32)
            if lip_points.size == 0:
                continue

            resized_file1 = replace_teeth_seamless(resized_file1, replacement_img, lip_points)

        # After teeth replacement, resize the processed image back to the original size
        final_img = cv2.resize(resized_file1, (original_size[1], original_size[0]))  # Resize to (width, height)

        # Save the final image
        save_path_final = r"static\images\final.jpg"
        cv2.imwrite(save_path_final, final_img)

        # Return the URLs for both images
        final_image_url = os.path.join(settings.STATIC_URL, 'images/final.jpg')
        file1_image_url = os.path.join(settings.STATIC_URL, 'images/file1.jpg')

        return Response({
            'before_image_url': file1_image_url,  # Original file1 image
            'after_image_url': final_image_url  # Final processed image
        }, status=status.HTTP_200_OK)
