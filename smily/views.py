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






# import requests
# from django.conf import settings
# from rest_framework.response import Response
# from rest_framework.decorators import api_view
# from rest_framework.views import APIView
# from rest_framework import status
# import cv2
# import numpy as np
# import face_alignment
# from PIL import Image, ImageEnhance
# import os


# class imageapi(APIView):

#     def post(self, request, *args, **kwargs):
#         # Check for file1 and file2 in request files
#         file1 = request.FILES.get('file1')
#         file2 = request.FILES.get('file2')  # This can either be a file or we'll expect a URL

#         # Log the request data for debugging
#         print(f"Request Data: {request.data}")

#         # Check if file1 is provided
#         if not file1:
#             return Response({"error": "File1 must be provided."}, status=status.HTTP_400_BAD_REQUEST)

#         # Handle file2 as a URL or file
#         file2_array = None
#         if not file2:
#             # Check if file2 is provided as a URL in the data
#             file2_url = request.data.get('file2')  # Expecting 'file2_url' in the request body
#             if not file2_url:
#                 return Response({"error": "File2 (either a file or a URL) must be provided."}, status=status.HTTP_400_BAD_REQUEST)

#             try:
#                 # Download the image from the provided URL
#                 response = requests.get(file2_url)
#                 if response.status_code == 200:
#                     file2_array = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
#                 else:
#                     return Response({"error": "Could not retrieve the image from the URL."}, status=status.HTTP_400_BAD_REQUEST)
#             except Exception as e:
#                 return Response({"error": f"Error downloading image from URL: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)
#         else:
#             # Convert the uploaded file2 to OpenCV array
#             try:
#                 file2_array = cv2.imdecode(np.frombuffer(file2.read(), np.uint8), cv2.IMREAD_COLOR)
#             except Exception as e:
#                 return Response({"error": f"Error reading file2: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)

#         # Convert file1 to OpenCV array
#         try:
#             file1_array = cv2.imdecode(np.frombuffer(file1.read(), np.uint8), cv2.IMREAD_COLOR)
#         except Exception as e:
#             return Response({"error": f"Error reading file1: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)

#         if file1_array is None or file2_array is None:
#             return Response({"error": "Could not decode one or both images."}, status=status.HTTP_400_BAD_REQUEST)

#         # Save the files
#         save_path1 = r"smily\images\file1.jpg"
#         save_path2 = r"smily\images\file2.jpg"
#         cv2.imwrite(save_path1, file1_array)
#         cv2.imwrite(save_path2, file2_array)
#         print("File1 and File2 saved successfully")

#         def resize_image(image, output_width, output_height):
#             return cv2.resize(image, (output_width, output_height))

#         # Initialize face alignment model
#         fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device='cpu')

#         # Load the input image (file1)
#         img = cv2.imread(save_path1)
#         if img is None:
#             return Response({'error': 'Could not load the input image.'}, status=status.HTTP_400_BAD_REQUEST)

#         output_width, output_height = 615, 500
#         img = resize_image(img, output_width, output_height)

#         # Convert the resized image to RGB
#         rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         # Detect landmarks
#         landmarks = fa.get_landmarks(rgb_img)
#         print(landmarks, "===========================landmarks")

#         # Load the replacement image (file2)
#         replacement_img = cv2.imread(save_path2)
#         if replacement_img is None:
#             return Response({'error': 'Could not load the replacement image.'}, status=status.HTTP_400_BAD_REQUEST)

#         def create_mask(lip_points, shape):
#             """Create a mask for the given lip points using convex hull."""
#             mask = np.zeros(shape, dtype=np.uint8)
#             hull = cv2.convexHull(lip_points)
#             cv2.fillConvexPoly(mask, hull, 255)
#             return mask

#         def blend_teeth(original_img, replacement_img, lip_points):
#             """Blend replacement teeth into the original image at the lip region."""
#             # Calculate the bounding box for the lip area
#             (x, y, w, h) = cv2.boundingRect(lip_points)

#             # Resize the replacement image to fit the bounding box
#             replacement_img_resized = cv2.resize(replacement_img, (w, h))

#             # Create a mask for the lip area within the bounding box
#             mask = create_mask(lip_points - [x, y], (h, w))

#             # Convert mask to 3-channel
#             mask_3channel = np.stack([mask] * 3, axis=-1)

#             # Extract the region of interest (ROI) from the original image
#             roi = original_img[y:y+h, x:x+w]

#             # Create inverse mask for blending
#             mask_inv = cv2.bitwise_not(mask)
#             mask_inv_3channel = np.stack([mask_inv] * 3, axis=-1)

#             # Create background and foreground images
#             img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
#             replacement_fg = cv2.bitwise_and(replacement_img_resized, replacement_img_resized, mask=mask)

#             # Combine the background and foreground
#             blended = cv2.add(img_bg, replacement_fg)

#             # Place the blended image back into the original image
#             original_img[y:y+h, x:x+w] = blended

#             return original_img

#         # Process landmarks and replace teeth
#         for landmark_set in landmarks:
#             lip_points = np.array([[int(landmark_set[i][0]), int(landmark_set[i][1])] for i in range(48, 68)], dtype=np.int32)
#             img = blend_teeth(img, replacement_img, lip_points)

#         # Adjust contrast and brightness
#         final_img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         final_img_pil = Image.fromarray(final_img_rgb)
#         contrast_enhancer = ImageEnhance.Contrast(final_img_pil)
#         final_img_pil = contrast_enhancer.enhance(1)
#         brightness_enhancer = ImageEnhance.Brightness(final_img_pil)
#         final_img_pil = brightness_enhancer.enhance(1)

#         # Convert back to OpenCV format
#         final_img = cv2.cvtColor(np.array(final_img_pil), cv2.COLOR_RGB2BGR)

#         # Save the final image
#         save_path_final = r"static\images\final.jpg"
#         cv2.imwrite(save_path_final, final_img)
#         print("Final image saved successfully.")

#         # Return the final image's URL
#         final_image_url = os.path.join(settings.STATIC_URL, 'images/final.jpg')
#         return Response({'final_image_url': final_image_url}, status=status.HTTP_200_OK)







import requests
from django.conf import settings
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework.views import APIView
from rest_framework import status
import cv2
import numpy as np
import face_alignment
from PIL import Image, ImageEnhance
import os

class imageapi(APIView):

    def post(self, request, *args, **kwargs):
        # Check for file1 and file2 in request files
        file1 = request.FILES.get('file1')
        file2 = request.FILES.get('file2')  # This can either be a file or we'll expect a URL

        # Check if file1 is provided
        if not file1:
            return Response({"error": "File1 must be provided."}, status=status.HTTP_400_BAD_REQUEST)

        # Handle file2 as a URL or file
        file2_array = None
        if not file2:
            # Check if file2 is provided as a URL in the data
            file2_url = request.data.get('file2')
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

        def resize_image(image, output_width, output_height):
            return cv2.resize(image, (output_width, output_height))

        # Initialize face alignment model
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device='cpu')

        # Load the input image (file1)
        img = cv2.imread(save_path1)
        if img is None:
            return Response({'error': 'Could not load the input image.'}, status=status.HTTP_400_BAD_REQUEST)

        output_width, output_height = 615, 500
        img = resize_image(img, output_width, output_height)

        # Convert the resized image to RGB
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect landmarks
        landmarks = fa.get_landmarks(rgb_img)

        if landmarks is None:
            return Response({'error': 'Could not detect facial landmarks.'}, status=status.HTTP_400_BAD_REQUEST)

        # Load the replacement image (file2)
        replacement_img = cv2.imread(save_path2)
        if replacement_img is None:
            return Response({'error': 'Could not load the replacement image.'}, status=status.HTTP_400_BAD_REQUEST)

        def create_mask(lip_points, shape):
            """Create a mask for the given lip points using convex hull."""
            mask = np.zeros(shape, dtype=np.uint8)
            hull = cv2.convexHull(lip_points)
            cv2.fillConvexPoly(mask, hull, 255)
            return mask

        def blend_teeth(original_img, replacement_img, lip_points):
            """Blend replacement teeth into the original image at the lip region."""
            (x, y, w, h) = cv2.boundingRect(lip_points)

            # Resize the replacement image to fit the bounding box
            replacement_img_resized = cv2.resize(replacement_img, (w, h))

            # Create a mask for the lip area
            mask = create_mask(lip_points - [x, y], (h, w))

            # Convert mask to 3-channel
            mask_3channel = np.stack([mask] * 3, axis=-1)

            # Extract the region of interest (ROI)
            roi = original_img[y:y+h, x:x+w]

            # Create inverse mask for blending
            mask_inv = cv2.bitwise_not(mask)
            mask_inv_3channel = np.stack([mask_inv] * 3, axis=-1)

            # Create background and foreground images
            img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            replacement_fg = cv2.bitwise_and(replacement_img_resized, replacement_img_resized, mask=mask)

            # Combine the background and foreground
            blended = cv2.add(img_bg, replacement_fg)

            # Place the blended image back into the original image
            original_img[y:y+h, x:x+w] = blended

            return original_img

        # Process landmarks and replace teeth
        for landmark_set in landmarks:
            lip_points = np.array([[int(landmark_set[i][0]), int(landmark_set[i][1])] for i in range(48, 68)], dtype=np.int32)
            img = blend_teeth(img, replacement_img, lip_points)

        # Adjust contrast and brightness
        final_img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        final_img_pil = Image.fromarray(final_img_rgb)
        contrast_enhancer = ImageEnhance.Contrast(final_img_pil)
        final_img_pil = contrast_enhancer.enhance(1.2)
        brightness_enhancer = ImageEnhance.Brightness(final_img_pil)
        final_img_pil = brightness_enhancer.enhance(1.1)

        # Convert back to OpenCV format
        final_img = cv2.cvtColor(np.array(final_img_pil), cv2.COLOR_RGB2BGR)

        # Save the final image
        save_path_final = r"static\images\final.jpg"
        cv2.imwrite(save_path_final, final_img)

        # Return the final image's URL
        final_image_url = os.path.join(settings.STATIC_URL, 'images/final.jpg')
        return Response({'final_image_url': final_image_url}, status=status.HTTP_200_OK)
















# import requests
# from django.conf import settings
# from rest_framework.response import Response
# from rest_framework.decorators import api_view
# from rest_framework.views import APIView
# from rest_framework import status
# import cv2
# import numpy as np
# import face_alignment
# from PIL import Image, ImageEnhance
# import os

# class imageapi(APIView):

#     def post(self, request, *args, **kwargs):
#         # Check for file1 and file2 in request files
#         file1 = request.FILES.get('file1')
#         file2 = request.FILES.get('file2')  # This can either be a file or we'll expect a URL

#         # Log the request data for debugging
#         print(f"Request Data: {request.data}")

#         # Check if file1 is provided
#         if not file1:
#             return Response({"error": "File1 must be provided."}, status=status.HTTP_400_BAD_REQUEST)

#         # Handle file2 as a URL or file
#         file2_array = None
#         if not file2:
#             # Check if file2 is provided as a URL in the data
#             file2_url = request.data.get('file2')  # Expecting 'file2_url' in the request body
#             if not file2_url:
#                 return Response({"error": "File2 (either a file or a URL) must be provided."}, status=status.HTTP_400_BAD_REQUEST)

#             try:
#                 # Download the image from the provided URL
#                 response = requests.get(file2_url)
#                 if response.status_code == 200:
#                     file2_array = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
#                 else:
#                     return Response({"error": "Could not retrieve the image from the URL."}, status=status.HTTP_400_BAD_REQUEST)
#             except Exception as e:
#                 return Response({"error": f"Error downloading image from URL: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)
#         else:
#             # Convert the uploaded file2 to OpenCV array
#             try:
#                 file2_array = cv2.imdecode(np.frombuffer(file2.read(), np.uint8), cv2.IMREAD_COLOR)
#             except Exception as e:
#                 return Response({"error": f"Error reading file2: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)

#         # Convert file1 to OpenCV array
#         try:
#             file1_array = cv2.imdecode(np.frombuffer(file1.read(), np.uint8), cv2.IMREAD_COLOR)
#         except Exception as e:
#             return Response({"error": f"Error reading file1: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)

#         if file1_array is None or file2_array is None:
#             return Response({"error": "Could not decode one or both images."}, status=status.HTTP_400_BAD_REQUEST)

#         # Save the files
#         save_path1 = r"smily\images\file1.jpg"
#         save_path2 = r"smily\images\file2.jpg"
#         cv2.imwrite(save_path1, file1_array)
#         cv2.imwrite(save_path2, file2_array)
#         print("File1 and File2 saved successfully")

#         def resize_image(image, output_width, output_height):
#             return cv2.resize(image, (output_width, output_height))

#         # Initialize face alignment model
#         fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device='cpu')

#         # Load the input image (file1)
#         img = cv2.imread(save_path1)
#         if img is None:
#             return Response({'error': 'Could not load the input image.'}, status=status.HTTP_400_BAD_REQUEST)

#         output_width, output_height = 615, 500
#         img = resize_image(img, output_width, output_height)

#         # Convert the resized image to RGB
#         rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         # Detect landmarks
#         landmarks = fa.get_landmarks(rgb_img)
#         print(landmarks, "===========================landmarks")

#         # Load the replacement image (file2)
#         replacement_img = cv2.imread(save_path2)
#         if replacement_img is None:
#             return Response({'error': 'Could not load the replacement image.'}, status=status.HTTP_400_BAD_REQUEST)

#         print("======================code here")

#         # Process landmarks and replace teeth
#         for landmark_set in landmarks:
#             lip_points = np.array([[int(landmark_set[i][0]), int(landmark_set[i][1])] for i in range(48, 68)], dtype=np.int32)
#             print("=====================landmark process")
#             (x, y, w, h) = cv2.boundingRect(lip_points)
#             print("===============================>")

#             print(f"Lip area coordinates: Top-left (x, y): ({x}, {y}), Width: {w}, Height: {h}")
#             padding = 3
#             x = max(x - padding, 0)
#             y = max(y - padding, 0)
#             w = min(w + 2 * padding, img.shape[1] - x)
#             h = min(h + 2 * padding, img.shape[0] - y)

#             replacement_img_resized = cv2.resize(replacement_img, (w, h))
#             mask = np.zeros((h, w), dtype=np.uint8)
#             cv2.fillConvexPoly(mask, lip_points - [x, y], 255)
#             roi = img[y:y+h, x:x+w]
#             img[y:y+h, x:x+w] = cv2.seamlessClone(replacement_img_resized, roi, mask, (w//2, h//2), cv2.NORMAL_CLONE)

#         # Adjust contrast and brightness
#         final_img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         final_img_pil = Image.fromarray(final_img_rgb)
#         contrast_enhancer = ImageEnhance.Contrast(final_img_pil)
#         final_img_pil = contrast_enhancer.enhance(1)
#         brightness_enhancer = ImageEnhance.Brightness(final_img_pil)
#         final_img_pil = brightness_enhancer.enhance(1)

#         # Convert back to OpenCV format
#         final_img = cv2.cvtColor(np.array(final_img_pil), cv2.COLOR_RGB2BGR)

#         # Save the final image
#         save_path_final = r"static\images\final.jpg"
#         cv2.imwrite(save_path_final, final_img)
#         print("Final image saved successfully.")

#         # Return the final image's URL
#         final_image_url = os.path.join(settings.STATIC_URL, 'images/final.jpg')
#         return Response({'final_image_url': final_image_url}, status=status.HTTP_200_OK)













# # from django.shortcuts import render
# # from rest_framework.decorators import api_view
# # from rest_framework.response import Response
# # from rest_framework import status
# # from django.http import JsonResponse,HttpResponse
# # import requests
# # from smily.models import ware
# # import face_alignment
# # from django.conf import settings
# # from matplotlib import pyplot as plt
# # import cv2
# # from rest_framework.views import APIView
# # import numpy as np
# # import os
# # from PIL import Image, ImageEnhance, ImageFilter
# # from django.core.files.storage import FileSystemStorage
# # # Create your views here.
# # class imageapi(APIView):
# #     def post(self, request, *args, **kwargs):
# #         file1 = request.FILES.get('file1')
# #         file2 = request.FILES.get('file2')  # This can either be a file or a URL

# #         # Check if file1 was provided
# #         if not file1:
# #             return Response({"error": "File1 must be provided."}, status=status.HTTP_400_BAD_REQUEST)

# #         # Handle file2 as a URL or file
# #         file2_array = None
# #         if file2 is None:
# #             # Check if file2 is a URL
# #             file2_url = request.data.get('file2_url')
# #             if not file2_url:
# #                 return Response({"error": "File2 (either a file or a URL) must be provided."}, status=status.HTTP_400_BAD_REQUEST)

# #             try:
# #                 # Download the image from the URL
# #                 response = requests.get(file2_url)
# #                 if response.status_code == 200:
# #                     file2_array = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
# #                 else:
# #                     return Response({"error": "Could not retrieve the image from the URL."}, status=status.HTTP_400_BAD_REQUEST)
# #             except Exception as e:
# #                 return Response({"error": f"Error downloading image from URL: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)
# #         else:
# #             # Convert the uploaded file2 to OpenCV array
# #             try:
# #                 file2_array = cv2.imdecode(np.frombuffer(file2.read(), np.uint8), cv2.IMREAD_COLOR)
# #             except Exception as e:
# #                 return Response({"error": f"Error reading file2: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)

# #         # Convert the uploaded file1 to OpenCV array
# #         try:
# #             file1_array = cv2.imdecode(np.frombuffer(file1.read(), np.uint8), cv2.IMREAD_COLOR)
# #         except Exception as e:
# #             return Response({"error": f"Error reading file1: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)

# #         if file1_array is None or file2_array is None:
# #             return Response({"error": "Could not decode one or both images."}, status=status.HTTP_400_BAD_REQUEST)

# #         # Save the files
# #         save_path1 = r"smily\images\file1.jpg"
# #         save_path2 = r"smily\images\file2.jpg"
# #         cv2.imwrite(save_path1, file1_array)
# #         cv2.imwrite(save_path2, file2_array)
# #         print("file1 and file2 saved successfully")

# #         def resize_image(image, output_width, output_height):
# #             return cv2.resize(image, (output_width, output_height))

# #         # Initialize face alignment model
# #         fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device='cpu')

# #         # Load the input image (file1)
# #         img = cv2.imread(save_path1)
# #         if img is None:
# #             return Response({'error': 'Could not load the input image.'}, status=status.HTTP_400_BAD_REQUEST)

# #         output_width, output_height = 615, 500
# #         img = resize_image(img, output_width, output_height)

# #         # Convert the resized image to RGB
# #         rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# #         # Detect landmarks
# #         landmarks = fa.get_landmarks(rgb_img)
# #         print(landmarks, "===========================landmarks")

# #         # Load the replacement image (file2)
# #         replacement_img = cv2.imread(save_path2)
# #         if replacement_img is None:
# #             return Response({'error': 'Could not load the replacement image.'}, status=status.HTTP_400_BAD_REQUEST)

# #         print("======================code here")

# #         # Process landmarks and replace teeth
# #         for landmark_set in landmarks:
# #             lip_points = np.array([[int(landmark_set[i][0]), int(landmark_set[i][1])] for i in range(48, 68)], dtype=np.int32)
# #             print("=====================landmark process")
# #             (x, y, w, h) = cv2.boundingRect(lip_points)
# #             print("===============================>")

# #             print(f"Lip area coordinates: Top-left (x, y): ({x}, {y}), Width: {w}, Height: {h}")
# #             padding = 3
# #             x = max(x - padding, 0)
# #             y = max(y - padding, 0)
# #             w = min(w + 2 * padding, img.shape[1] - x)
# #             h = min(h + 2 * padding, img.shape[0] - y)

# #             replacement_img_resized = cv2.resize(replacement_img, (w, h))
# #             mask = np.zeros((h, w), dtype=np.uint8)
# #             cv2.fillConvexPoly(mask, lip_points - [x, y], 255)
# #             roi = img[y:y+h, x:x+w]
# #             img[y:y+h, x:x+w] = cv2.seamlessClone(replacement_img_resized, roi, mask, (w//2, h//2), cv2.NORMAL_CLONE)

# #         # Adjust contrast and brightness
# #         final_img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# #         final_img_pil = Image.fromarray(final_img_rgb)
# #         contrast_enhancer = ImageEnhance.Contrast(final_img_pil)
# #         final_img_pil = contrast_enhancer.enhance(1)
# #         brightness_enhancer = ImageEnhance.Brightness(final_img_pil)
# #         final_img_pil = brightness_enhancer.enhance(1)

# #         # Convert back to OpenCV format
# #         final_img = cv2.cvtColor(np.array(final_img_pil), cv2.COLOR_RGB2BGR)

# #         # Save the final image
# #         save_path_final = r"static\images\final.jpg"
# #         cv2.imwrite(save_path_final, final_img)
# #         print("Final image saved successfully.")

# #         # Return the final image's URL
# #         final_image_url = os.path.join(settings.STATIC_URL, 'images/final.jpg')
# #         return Response({'final_image_url': final_image_url}, status=status.HTTP_200_OK)
#     # def post(self, request, *args, **kwargs):
#     #     file1 = request.FILES.get('file1')
#     #     file2 = request.FILES.get('file2')
#     #     # image_path = (r"smilefy\smily\images")
#     #     # print(image_path)
#     #     file1_array = cv2.imdecode(np.frombuffer(file1.read(), np.uint8), cv2.IMREAD_COLOR)
#     #     file2_array = cv2.imdecode(np.frombuffer(file2.read(), np.uint8), cv2.IMREAD_COLOR)
#     #     # fs = FileSystemStorage()
#     #     save_path1 =(r"smily\images\file1.jpg")
#     #     save_path2 =(r"smily\images\file2.jpg")
#     #     cv2.imwrite(save_path1, file1_array)
#     #     cv2.imwrite(save_path2, file2_array)
#     #     print("file1 is saved")
#     #     print("file1 is saved")
#     #     # Define the save path and filename
#     #     # filename1 = 'file1.jpg'
#     #     # filename2 = 'file2.jpg'
#     #     # file_path1 = fs.save(f'uploaded_images/{filename1}',file1_array)
        
        
        
#     #     # # file2_array = cv2.imdecode(np.frombuffer(file2.read(), np.uint8), cv2.IMREAD_COLOR)
#     #     # file_path2 = fs.save(f'uploaded_images/{filename2}',file2_array)

#     #     # print("here the code ")
#     #     # # Get the full file path where the image was saved
#     #     # full_file_path1 = fs.path(file_path1)
#     #     # full_file_path2 = fs.path(file_path2)

#     #     # print(f"Image saved to {full_file_path1},{full_file_path2}")
        
#     #     if not file1 or not file2:
#     #         return Response({"error": "Both images must be provided."}, status=status.HTTP_400_BAD_REQUEST)
        
#     #     # file1 = 'smilefy\smily\images\uploaded_images\file1.jpg'
        
#     #     # Convert uploaded files to OpenCV format
        
        
#     #     # file1_array = cv2.imdecode(np.frombuffer(file1.read(), np.uint8), cv2.IMREAD_COLOR)
#     #     # file2_array = cv2.imdecode(np.frombuffer(file2.read(), np.uint8), cv2.IMREAD_COLOR)

#     #     # # Save the uploaded files
#     #     # print(file1)
#     #     # print(file2)
#     #     # save_path1 = (r"smilefy\smily\images\uploaded_images\file1.jpg")
        
#     #     # save_path2 = (r"smilefy\smily\images\uploaded_images\file2.jpg")
#     #     # cv2.imwrite(save_path1, file1_array)
#     #     # cv2.imwrite(save_path2, file2_array)

#     #     def resize_image(image, output_width, output_height):
#     #         return cv2.resize(image, (output_width, output_height))

#     #     # Initialize face alignment model
#     #     fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device='cpu')

#     #     # Load the input image
#     #     img = cv2.imread(r"smily\images\file1.jpg")
        
#     #     if img is None:
#     #         return Response({'error': 'Could not load the input image.'}, status=status.HTTP_400_BAD_REQUEST)

#     #     output_width, output_height = 615, 500
#     #     img = resize_image(img, output_width, output_height)

#     #     # Convert the resized image to RGB
#     #     rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     #     # Detect landmarks
#     #     landmarks = fa.get_landmarks(rgb_img)
#     #     print(landmarks,"===========================landmarks")

#     #     # Load the replacement image
#     #     replacement_img = cv2.imread(r"smily\images\file2.jpg")
#     #     if replacement_img is None:
#     #         return Response({'error': 'Could not load the replacement image.'}, status=status.HTTP_400_BAD_REQUEST)
#     #     print("======================code yaha")
        
#     #     # Process landmarks and replace teeth
#     #     for landmark_set in landmarks:
#     #         lip_points = np.array([[int(landmark_set[i][0]), int(landmark_set[i][1])] for i in range(48, 68)], dtype=np.int32)
#     #         print("=====================13215465")
#     #         (x, y, w, h) = cv2.boundingRect(lip_points)
#     #         print("===============================>")

#     #         print(f"Lip area coordinates: Top-left (x, y): ({x}, {y}), Width: {w}, Height: {h}")
#     #         padding = 3
#     #         x = max(x - padding, 0)
#     #         y = max(y - padding, 0)
#     #         w = min(w + 2 * padding, img.shape[1] - x)
#     #         h = min(h + 2 * padding, img.shape[0] - y)

#     #         replacement_img_resized = cv2.resize(replacement_img, (w, h))
#     #         mask = np.zeros((h, w), dtype=np.uint8)
#     #         cv2.fillConvexPoly(mask, lip_points - [x, y], 255)
#     #         roi = img[y:y+h, x:x+w]
#     #         img[y:y+h, x:x+w] = cv2.seamlessClone(replacement_img_resized, roi, mask, (w//2, h//2), cv2.NORMAL_CLONE)

#     #     # Adjust contrast and brightness
#     #     final_img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     #     final_img_pil = Image.fromarray(final_img_rgb)
#     #     contrast_enhancer = ImageEnhance.Contrast(final_img_pil)
#     #     final_img_pil = contrast_enhancer.enhance(1)
#     #     brightness_enhancer = ImageEnhance.Brightness(final_img_pil)
#     #     final_img_pil = brightness_enhancer.enhance(1)

#     #     # Convert back to OpenCV format
#     #     final_img = cv2.cvtColor(np.array(final_img_pil), cv2.COLOR_RGB2BGR)

#     #     # Save the final image
#     #     save_path_final = (r"static\images\final.jpg")
#     #     cv2.imwrite(save_path_final, final_img)
#     #     print("final images saved ")

#     #     # Return the final image's URL
#     #     final_image = os.path.join(settings.STATIC_URL, r'images/final.jpg')
#     #     return Response({'final_image_url': final_image}, status=status.HTTP_200_OK)
#     # def post(self, request, *args, **kwargs):
#     #     # Get the uploaded file1
#     #     file1 = request.FILES.get('file1')
#     #     file2 = request.FILES.get('file2')  # This can either be a file or a URL

#     #     # Check if file1 was provided
#     #     if not file1:
#     #         return Response({"error": "File1 must be provided."}, status=status.HTTP_400_BAD_REQUEST)

#     #     # Handle file2 as a URL or file
#     #     file2_array = None
#     #     if file2 is None:
#     #         # Check if file2 is a URL
#     #         file2_url = request.data.get('file2_url')
#     #         if not file2_url:
#     #             return Response({"error": "File2 (either a file or a URL) must be provided."}, status=status.HTTP_400_BAD_REQUEST)

#     #         try:
#     #             # Download the image from the URL
#     #             response = requests.get(file2_url)
#     #             if response.status_code == 200:
#     #                 file2_array = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
#     #             else:
#     #                 return Response({"error": "Could not retrieve the image from the URL."}, status=status.HTTP_400_BAD_REQUEST)
#     #         except Exception as e:
#     #             return Response({"error": f"Error downloading image from URL: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)
#     #     else:
#     #         # Convert the uploaded file2 to OpenCV array
#     #         try:
#     #             file2_array = cv2.imdecode(np.frombuffer(file2.read(), np.uint8), cv2.IMREAD_COLOR)
#     #         except Exception as e:
#     #             return Response({"error": f"Error reading file2: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)

#     #     # Convert the uploaded file1 to OpenCV array
#     #     try:
#     #         file1_array = cv2.imdecode(np.frombuffer(file1.read(), np.uint8), cv2.IMREAD_COLOR)
#     #     except Exception as e:
#     #         return Response({"error": f"Error reading file1: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)

#     #     if file1_array is None or file2_array is None:
#     #         return Response({"error": "Could not decode one or both images."}, status=status.HTTP_400_BAD_REQUEST)

#     #     # Save the files
#     #     save_path1 = r"smily\images\file1.jpg"
#     #     save_path2 = r"smily\images\file2.jpg"
#     #     cv2.imwrite(save_path1, file1_array)
#     #     cv2.imwrite(save_path2, file2_array)
#     #     print("File 1 and File 2 saved successfully.")

#     #     # Rest of the code for processing images and face alignment...
#     #     # Initialize face alignment model
#     #     fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device='cpu')

#     #     # Load the input image
#     #     img = cv2.imread(save_path1)
#     #     if img is None:
#     #         return Response({'error': 'Could not load the input image.'}, status=status.HTTP_400_BAD_REQUEST)

#     #     # Resize the image and perform landmark detection
#     #     output_width, output_height = 615, 500
#     #     img = cv2.resize(img, (output_width, output_height))
#     #     rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     #     landmarks = fa.get_landmarks(rgb_img)

#     #     if landmarks is None:
#     #         return Response({'error': 'No landmarks detected.'}, status=status.HTTP_400_BAD_REQUEST)

#     #     # Load the replacement image and continue processing...
#     #     replacement_img = cv2.imread(save_path2)
#     #     if replacement_img is None:
#     #         return Response({'error': 'Could not load the replacement image.'}, status=status.HTTP_400_BAD_REQUEST)

#     #     # The rest of your processing logic...
        
#     #     # Save and return the final image URL
#     #     save_path_final = r"static\images\final.jpg"
#     #     cv2.imwrite(save_path_final, img)
#     #     final_image_url = os.path.join(settings.STATIC_URL, 'images/final.jpg')
#     #     return Response({'final_image_url': final_image_url}, status=status.HTTP_200_OK)
