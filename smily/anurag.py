def home(request):
    if request.method == 'POST':
        file1= request.FILES.get('file1')
        file2 = request.FILES.get('file2')
        
        file1_array = cv2.imdecode(np.frombuffer(file1.read(), np.uint8), cv2.IMREAD_COLOR)
        save_path1 =(r"smily\images\file1.jpg")
        cv2.imwrite(save_path1, file1_array)
        print("file1 is saved")
        
        if file2:
            file2_array = cv2.imdecode(np.frombuffer(file2.read(), np.uint8), cv2.IMREAD_COLOR)
            save_path2 = (r"smily\images\file2.jpg")
            cv2.imwrite(save_path2, file2_array)
            print("file2 is saved")
    
        # print("=============>",file1)
        # # print("==========>",file2)
        # model = ware.objects.create(file1=file1_array)
        # model = ware.objects.create(file2=file2_array)
        # model.save()
        
        def resize_image(image, output_width, output_height):
            return cv2.resize(image, (output_width, output_height))

        # Initialize face alignment model
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False,device='cpu')

        # Load the input image
        print("Code goes here")
        img = cv2.imread(r"smily\images\file1.jpg")
        print("Code goes here !!!!")
        
        if img is None:
            print("Error: Could not load image. Check the file path.")
        else:
            output_width, output_height = 615, 500
            img = resize_image(img, output_width, output_height)

            # Convert the resized image to RGB
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Detect landmarks
            landmarks = fa.get_landmarks(rgb_img)

            # Load the replacement image
            replacement_img = cv2.imread(r"smily\images\file2.jpg")
            if replacement_img is None:
                print("Error: Could not load replacement image.")
            else:
                # Loop over detected landmarks
                for landmark_set in landmarks:
                    # Extract the coordinates for the lip area (approximate teeth region)
                    lip_points = np.array([[int(landmark_set[i][0]), int(landmark_set[i][1])] for i in range(48, 68)], dtype=np.int32)

                    # Find the bounding box for the lip area
                    (x, y, w, h) = cv2.boundingRect(lip_points)
                    print(f"Lip area coordinates: Top-left (x, y): ({x}, {y}), Width: {w}, Height: {h}")

                    padding = 3  # Adjust this value as needed
                    x = max(x - padding, 0)
                    y = max(y - padding, 0)
                    w = min(w + 2 * padding, img.shape[1] - x)
                    h = min(h + 2 * padding, img.shape[0] - y)

                    # Resize replacement image to fit the bounding box dimensions
                    replacement_img_resized = cv2.resize(replacement_img, (w, h))

                    # Create a mask for the replacement image
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillConvexPoly(mask, lip_points - [x, y], 255)

                    # Create a region of interest (ROI) in the original image
                    roi = img[y:y+h, x:x+w]

                    # Blend the replacement image onto the original image using seamless cloning
                    img[y:y+h, x:x+w] = cv2.seamlessClone(replacement_img_resized, roi, mask, (w//2, h//2), cv2.NORMAL_CLONE)

                    # Optional: Draw the landmarks and bounding box for visualization
                    # cv2.polylines(img, [lip_points], isClosed=True, color=(0, 255, 0), thickness=1)

                # Convert the image from BGR to RGB for PIL compatibility
                final_img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                final_img_pil = Image.fromarray(final_img_rgb)

                # Step 3: Adjust Contrast
                contrast_enhancer = ImageEnhance.Contrast(final_img_pil)
                final_img_pil = contrast_enhancer.enhance(1)  # Adjust the contrast factor as needed

                # Step 4: Adjust Brightness
                brightness_enhancer = ImageEnhance.Brightness(final_img_pil)
                final_img_pil = brightness_enhancer.enhance(1)  # Adjust the brightness factor as needed

                # Convert back to OpenCV format for final processing
                final_img = cv2.cvtColor(np.array(final_img_pil), cv2.COLOR_RGB2BGR)
