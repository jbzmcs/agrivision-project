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
    (h, w) = image.shape[:2]
    new_width = 775
    aspect_ratio = h / w
    new_height = int(new_width * aspect_ratio)
    resized_image = cv2.resize(image, (new_width, new_height))

    # --- 2. Blur & Convert ---
    blurred_image = cv2.GaussianBlur(resized_image, (7, 7), 0)
    hsv_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)

    # --- 3. Create Leaf Mask ---
    s_channel = hsv_image[:, :, 1]
    _, leaf_mask = cv2.threshold(s_channel, 40, 255, cv2.THRESH_BINARY)

    # --- 4. Create Disease Mask ---
    lower_disease_range = np.array([10, 50, 50])
    upper_disease_range = np.array([35, 255, 255])
    disease_mask = cv2.inRange(hsv_image, lower_disease_range, upper_disease_range)

    # --- 5. Isolate Spots on Leaf ---
    final_spot_mask = cv2.bitwise_and(disease_mask, leaf_mask)

    # --- 6. Calculate Features ---
    leaf_area = cv2.countNonZero(leaf_mask)
    spot_area = cv2.countNonZero(final_spot_mask)

    (cnts, _) = cv2.findContours(final_spot_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    spot_count = len(cnts)

    # --- 7. Calculate Severity ---
    severity_percentage = (spot_area / leaf_area) * 100 if leaf_area > 0 else 0

    return [severity_percentage, spot_count]


def run_feature_extraction():
    """
    Loops through all datasets (train, val, test),
    extracts features, and saves to a single CSV file.
    """
    DATA_FOLDERS = ["data/train", "data/val", "data/test"]
    all_features_data = []
    start_time = time.time()

    for DATA_FOLDER in DATA_FOLDERS:
        try:
            CLASSES = [d for d in os.listdir(DATA_FOLDER) if os.path.isdir(os.path.join(DATA_FOLDER, d))]
            if not CLASSES:
                print(f"  No class subdirectories found in {DATA_FOLDER}. Skipping.")
                continue
        except FileNotFoundError:
            print(f" Error: Data folder not found at {DATA_FOLDER}")
            continue

        print(f"\n Starting feature extraction from: {DATA_FOLDER}")
        print(f"   Found classes: {CLASSES}")

        for plant_class in CLASSES:
            class_path = os.path.join(DATA_FOLDER, plant_class)
            image_files = os.listdir(class_path)
            print(f"   Processing {len(image_files)} images in class: {plant_class}")

            for image_name in image_files:
                image_path = os.path.join(class_path, image_name)

                if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    print(f"   Skipping non-image file: {image_name}")
                    continue

                features = extract_features(image_path)
                if features is not None:
                    features.append(plant_class)
                    all_features_data.append(features)

    # --- Save the results ---
    if not all_features_data:
        print("\nNo images were processed. CSV file will be empty.")
        return

    print("\n----------------------------------")
    print("Feature extraction complete!")
    print(f"   Total images processed: {len(all_features_data)}")
    print(f"   Time elapsed: {time.time() - start_time:.2f} seconds")

    # --- Create CSV ---
    column_names = ["severity_percent", "spot_count", "class_label"]
    df = pd.DataFrame(all_features_data, columns=column_names)

    output_csv_path = "train_features.csv"
    df.to_csv(output_csv_path, index=False)

    print(f"\n Successfully created '{output_csv_path}'")
    print("\n Here's a preview of your dataset:")
    print(df.head())


if __name__ == "__main__":
    run_feature_extraction()
