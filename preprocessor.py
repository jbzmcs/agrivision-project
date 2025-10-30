import cv2
import numpy as np
import os
import pandas as pd
import time

def extract_features(image_path):
    """
    Reads an image, processes it with OpenCV, and extracts
    disease severity percentage and spot count.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read image {image_path}. Skipping.")
        return None 

    # --- 1. Resize ---
    # Resize image while maintaining aspect ratio
    (h, w) = image.shape[:2]
    new_width = 775 
    aspect_ratio = h / w
    new_height = int(new_width * aspect_ratio)
    resized_image = cv2.resize(image, (new_width, new_height))

    # --- 2. Blur & Convert ---
    # Apply Gaussian blur and convert to HSV color space
    blurred_image = cv2.GaussianBlur(resized_image, (7, 7), 0)
    hsv_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)
    
    # --- 3. Create Leaf Mask (Teammate's Method) ---
    # NOTE: This method assumes the background is less saturated than the leaf.
    # This is a potential point of failure.
    s_channel = hsv_image[:, :, 1]
    _, leaf_mask = cv2.threshold(s_channel, 40, 255, cv2.THRESH_BINARY)
    
    # --- 4. Create Disease Mask ---
    # This range targets yellow/light brown spots in the HSV space
    lower_disease_range = np.array([10, 50, 50])
    upper_disease_range = np.array([35, 255, 255])
    disease_mask = cv2.inRange(hsv_image, lower_disease_range, upper_disease_range)
    
    # --- 5. Isolate Spots on Leaf (Excellent!) ---
    # Use bitwise_and to find spots that are ONLY on the leaf.
    final_spot_mask = cv2.bitwise_and(disease_mask, leaf_mask)
    
    # --- 6. Calculate Features ---
    # Get the total pixel area for the leaf and the spots
    leaf_area = cv2.countNonZero(leaf_mask)
    spot_area = cv2.countNonZero(final_spot_mask)
    
    # Find contours to count the number of distinct spots
    (cnts, _) = cv2.findContours(final_spot_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    spot_count = len(cnts) 

    # --- 7. Calculate Severity ---
    # Handle division-by-zero case
    if leaf_area > 0:
        severity_percentage = (spot_area / leaf_area) * 100 
    else:
        severity_percentage = 0
        
    return [severity_percentage, spot_count]

def run_feature_extraction():
    """
    Main execution function to loop through data, extract features,
    and save to a CSV file.
    """

    DATA_FOLDER = "data/train"

    try:
        CLASSES = [d for d in os.listdir(DATA_FOLDER) if os.path.isdir(os.path.join(DATA_FOLDER, d))]
        if not CLASSES:
            print(f"Error: No class subdirectories found in {DATA_FOLDER}.")
            print("Please ensure your 'data/train' folder is populated, e.g., 'data/train/Tomato_Healthy'")
            return
    except FileNotFoundError:
        print(f"Error: Data folder not found at {DATA_FOLDER}")
        print(f"Please make sure you are running this script from the 'agrivision-project' root folder.")
        return

    start_time = time.time()
    all_features_data = [] 

    print(f"Starting feature extraction from: {DATA_FOLDER}")
    print(f"Found classes: {CLASSES}")

    for plant_class in CLASSES:
        class_path = os.path.join(DATA_FOLDER, plant_class)
        
        image_files = os.listdir(class_path)
        print(f"\nProcessing {len(image_files)} images in class: {plant_class}")

        for image_name in image_files:
            image_path = os.path.join(class_path, image_name)
            
            # Filter out non-image files
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                print(f"Skipping non-image file: {image_name}")
                continue
                
            features = extract_features(image_path)
            
            if features is not None:
                # Add the class label to the features list
                features.append(plant_class)
                all_features_data.append(features)

    if not all_features_data:
        print("\nError: No images were processed. Output CSV will be empty.")
        return

    print("\n----------------------------------")
    print("Feature extraction complete!")
    print(f"Processed a total of {len(all_features_data)} images in {time.time() - start_time:.2f} seconds.")

    # --- 8. Save to CSV ---
    column_names = ["severity_percent", "spot_count", "class_label"]
    df = pd.DataFrame(all_features_data, columns=column_names)

    output_csv_path = "train_features.csv"
    df.to_csv(output_csv_path, index=False)

    print(f"Successfully created '{output_csv_path}'")
    print("\nHere's a sample of your new dataset:")
    print(df.head()) 


if __name__ == "__main__":
    run_feature_extraction()