import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split # Import at the top as per best practices

# --- Configuration and Data Loading ---
# Define the path to your dataset.
# IMPORTANT: Adjust this path if your dataset is not located here
# when running on a different machine or if your project structure changes.
directory = "D:\\Computer vision\\Files\\Datasets\\homer_bart_1\\homer_bart_1" 

# Collect all image file paths
files = [os.path.join(directory, f) for f in sorted(os.listdir(directory))]

# Initialize the header for the CSV export file
export = 'mouth_percent,pants_percent,shoes_percent,tshirt_percent,shorts_percent,sneakers_percent,class_label\n'

# Flag to control image display during processing
show_images = True 
# List to store features extracted from each image
features = []

# Flag to allow controlled exit from the main processing loop
should_exit_loop = False 

# --- Main Image Processing Loop ---
for image_path in files:
    # Check if the user wants to exit the loop
    if should_exit_loop: 
        break

    try:
        # Read the image in BGR format (default for OpenCV)
        original_image = cv2.imread(image_path)
        if original_image is None: # Check if image was loaded successfully
            print(f"Warning: Could not read image at {image_path}. Skipping.")
            continue
        (H, W) = original_image.shape[:2] # Get image dimensions
    except Exception as e:
        print(f"Error processing {image_path}: {e}. Skipping.")
        continue

    # Create a copy of the image to draw on, keeping the original intact
    image = original_image.copy()
    image_features = [] # Features for the current image

    # Initialize counters for detected colored pixels
    mouth = pants = shoes = 0
    tshirt = shorts = sneakers = 0 

    # Determine character class based on file name prefix
    # 'b' for Bart, anything else (assumed) for Homer
    image_name = os.path.basename(os.path.normpath(image_path))
    if image_name.startswith('b'):
        class_name = 0 # Bart
    else:
        class_name = 1 # Homer

    # Iterate over each pixel in the image
    for height in range(0, H):
        for width in range(0, W):
            # Get BGR color components of the current pixel
            blue = image.item(height, width, 0)
            green = image.item(height, width, 1)
            red = image.item(height, width, 2)
            
            # --- Character-Specific Color Detection Logic ---
            if class_name == 1: # Logic for Homer
                # Homer - brown mouth detection
                # This range might need fine-tuning based on your dataset's specific brown shades
                if (80 <= blue <= 100 and 80 <= green <= 100 and 80 <= red <= 100):
                    image[height, width] = [0, 255, 255] # Change to Yellow (BGR)
                    mouth += 1
                # Homer - blue pants detection
                # This range might need fine-tuning
                if (150 <= blue <= 180 and 98 <= green <= 120 and 0 <= red <= 90):
                    image[height, width] = [0, 255, 255] # Change to Yellow (BGR)
                    pants += 1
                # Homer - gray shoes detection (only in the bottom half of the image)
                # This range might need fine-tuning
                if height > (H / 2): 
                    if (25 <= blue <= 45 and 25 <= green <= 45 and 25 <= red <= 45):
                        image[height, width] = [0, 255, 255] # Change to Yellow (BGR)
                        shoes += 1
            
            elif class_name == 0: # Logic for Bart
                # Bart - blue shorts detection
                # IMPORTANT: This range is an example. You MUST fine-tune it based on your actual Bart images.
                # If Bart's shorts were turning yellow previously by Homer's logic,
                # it means Homer's blue pants range was overlapping.
                # Adjust ranges carefully for both characters to avoid overlap.
                if (100 <= blue <= 150 and 0 <= green <= 50 and 0 <= red <= 50): # Example range for Bart's blue shorts
                    image[height, width] = [0, 255, 255] # Change to Yellow (BGR)
                    shorts += 1
                
                # TODO: Add logic here for Bart's orange/red t-shirt and white/red sneakers.
                # Example structure:
                # if (Bart's T-shirt color range):
                #     image[height, width] = [0, 255, 255] # Or any other highlight color
                #     tshirt += 1
                # if (Bart's Sneakers color range and height > (H / 2)):
                #     image[height, width] = [0, 255, 255] # Or any other highlight color
                #     sneakers += 1

    # --- Feature Calculation (Percentage of pixels) ---
    total_pixels = H * W
    mouth_percent = (mouth / total_pixels) * 100
    pants_percent = (pants / total_pixels) * 100
    shoes_percent = (shoes / total_pixels) * 100
    shorts_percent = (shorts / total_pixels) * 100 
    tshirt_percent = (tshirt / total_pixels) * 100 # Will be 0 until implemented for Bart
    sneakers_percent = (sneakers / total_pixels) * 100 # Will be 0 until implemented for Bart

    # Append calculated features to the list
    image_features.append(round(mouth_percent, 2))  # 0
    image_features.append(round(pants_percent, 2))  # 1
    image_features.append(round(shoes_percent, 2))  # 2
    image_features.append(round(tshirt_percent, 2)) # 3 (Currently 0 for Bart unless implemented)
    image_features.append(round(shorts_percent, 2)) # 4 
    image_features.append(round(sneakers_percent, 2)) # 5 (Currently 0 for Bart unless implemented)
    image_features.append(class_name)              # 6 -> Final label (0 for Bart, 1 for Homer)
    features.append(image_features)

    # Format features for CSV export
    f = (",".join([str(item) for item in image_features]))
    export += f + '\n'

    # --- Image Display and User Interaction ---
    if show_images == True:
        # Display images using OpenCV's imshow (expects BGR format)
        cv2.imshow("Original Image", original_image)
        cv2.imshow("Processed Image", image)
        
        key = cv2.waitKey(0) # Wait indefinitely until a key is pressed (0 means infinite wait)
        
        if key == ord('q') or key == 27: # 'q' key or 'Esc' key (ASCII 27)
            should_exit_loop = True # Set flag to exit the main loop
            cv2.destroyAllWindows() # Close all OpenCV display windows
        elif key == ord('0'): # '0' key
            cv2.destroyAllWindows() # Close windows and proceed to the next image
            # No explicit 'continue' needed here, as the loop naturally proceeds.
        else: # For any other key pressed (e.g., accidental press)
            cv2.destroyAllWindows()


# --- Post-Loop Cleanup and Data Export ---
# Ensure all OpenCV windows are closed after the main loop finishes
cv2.destroyAllWindows() 

# Write extracted features to a CSV file
with open('features.csv', 'w') as file:
    file.write(export)
# file.closed # 'with open' automatically handles closing the file

# --- Load and Prepare Dataset for Machine Learning ---
# Load the generated features.csv into a pandas DataFrame
dataset = pd.read_csv('features.csv')

# Separate features (X) and target labels (y)
# X includes mouth_percent to sneakers_percent (columns 0 to 5)
x = dataset.iloc[:, 0:6].values
# y is the class_label (column 6)
y = dataset.iloc[:, 6].values

# Split the dataset into training and testing sets
# 20% of data for testing, 80% for training
# random_state ensures reproducibility of the split
x_train, x_test, ytrain, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

print("Data loaded and split successfully!")
print(f"X_train shape: {x_train.shape}")
print(f"Y_train shape: {ytrain.shape}")