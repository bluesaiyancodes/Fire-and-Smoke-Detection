import cv2
import numpy as np
import os

def create_fire_map(frame, gmm_a, gmm_b):
    # Convert the frame to Lab color space
    lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
    
    # Split the Lab channels
    l, a, b = cv2.split(lab_frame)
    
    # Gaussian Mixture Model on a and b channels
    fg_mask_a = gmm_a.apply(a)
    fg_mask_b = gmm_b.apply(b)
    
    fire_map = cv2.bitwise_and(fg_mask_a, fg_mask_b)
    
    # Normalize and apply the backprojection technique
    fire_map = cv2.normalize(fire_map, None, 0, 255, cv2.NORM_MINMAX)
    
    return fire_map

def create_smoke_map(prev_frame, frame):
    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Frame difference
    smoke_map = cv2.absdiff(prev_gray, curr_gray)
    _, smoke_map = cv2.threshold(smoke_map, 50, 255, cv2.THRESH_BINARY)
    
    return smoke_map

def combined_detection(frame, mog2):
    # Apply BackgroundSubtractorMOG2
    fg_mask = mog2.apply(frame)
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    
    return fg_mask

def post_process(mask):
    # Morphological operations to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask

def save_image(output_dir, filename, image):
    cv2.imwrite(os.path.join(output_dir, filename), image)

def process_images_in_folder(folder_path, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get a sorted list of image files in the directory
    image_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('jpg', 'png', 'jpeg'))])
    
    if len(image_files) < 2:
        print("Not enough images in the folder for processing.")
        return
    
    # Initialize BackgroundSubtractorMOG2 with a short history
    mog2 = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold=16, detectShadows=False)
    
    # Initialize GMM for a and b channels
    gmm_a = cv2.createBackgroundSubtractorMOG2(history=5, varThreshold=16, detectShadows=False)
    gmm_b = cv2.createBackgroundSubtractorMOG2(history=5, varThreshold=16, detectShadows=False)
    
    # Read the first image to start processing
    prev_frame = cv2.imread(image_files[0])
    
    for idx, image_file in enumerate(image_files[1:], start=1):
        frame = cv2.imread(image_file)
        
        if frame is None or prev_frame is None:
            print(f"Skipping frame {image_file} due to loading issues.")
            continue
        
        # Fire map using Lab color model and GMM
        fire_map = create_fire_map(frame, gmm_a, gmm_b)
        
        # Smoke map using frame difference
        smoke_map = create_smoke_map(prev_frame, frame)
        
        # Combined detection using MOG2
        combined_map = combined_detection(frame, mog2)
        
        # Post-process the fire and smoke maps
        fire_map = post_process(fire_map)
        smoke_map = post_process(smoke_map)
        combined_map = post_process(combined_map)
        
        # Save the results
        save_image(output_dir, f"fire_map_{idx:03d}.png", fire_map)
        save_image(output_dir, f"smoke_map_{idx:03d}.png", smoke_map)
        save_image(output_dir, f"combined_map_{idx:03d}.png", combined_map)
        
        prev_frame = frame.copy()
    
    print(f"Processed images saved in {output_dir}")

if __name__ == "__main__":
    folder_path = "Dataset/fire/"  
    output_dir = "output/"  
    process_images_in_folder(folder_path, output_dir)
