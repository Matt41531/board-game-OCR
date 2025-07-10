import cv2
import numpy as np
import pytesseract
from matplotlib import pyplot as plt

def show_image(title, img):
    """Helper function to display images during development"""
    plt.figure(figsize=(10, 8))
    if len(img.shape) == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def preprocess_image(image_path):
    """Load and preprocess the image"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Resize for consistency while preserving aspect ratio
    height, width = img.shape[:2]
    max_dimension = 1000
    scale = max_dimension / max(height, width)
    new_width = int(width * scale)
    new_height = int(height * scale)
    img = cv2.resize(img, (new_width, new_height))
    
    # Convert to grayscale for feature detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray

def detect_victory_points(img):
    """Detect victory points (number next to feather icon on left side)"""
    # The victory points are always on the left side
    height, width = img.shape[:2]
    
    # Extract the left region where victory points are located
    left_region = img[int(height*0.2):int(height*0.4), 0:int(width*0.15)]
    
    # Convert to grayscale and apply thresholding
    gray = cv2.cvtColor(left_region, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    
    # Use morphological operations to clean up the image
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Use OCR to extract the number
    config = '--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789'
    vp_text = pytesseract.image_to_string(thresh, config=config).strip()
    
    # If OCR fails, try to find contours and analyze them
    if not vp_text.isdigit():
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            x, y, w, h = cv2.boundingRect(contour)
            if 0.5 < h/w < 2.5:  # Reasonable aspect ratio for a digit
                digit_roi = thresh[y:y+h, x:x+w]
                vp_text = pytesseract.image_to_string(digit_roi, config=config).strip()
                if vp_text.isdigit():
                    break
    
    return vp_text if vp_text.isdigit() else "Unknown"

def detect_egg_capacity(img):
    """Detect egg capacity (number of egg icons on left side)"""
    height, width = img.shape[:2]
    
    # Extract the left region where egg icons are located
    left_region = img[int(height*0.4):int(height*0.6), 0:int(width*0.15)]
    
    # Convert to grayscale and apply thresholding
    gray = cv2.cvtColor(left_region, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    # Use morphological operations to clean up the image
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Find contours of egg shapes
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size and shape to identify eggs
    egg_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:  # Minimum area threshold
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            # Egg icons typically have an aspect ratio close to 1.5-2.0
            if 1.0 < aspect_ratio < 3.0:
                egg_contours.append(contour)
    
    # Count the number of egg icons (divide by 2 because we might detect both inner and outer contours)
    egg_count = max(1, len(egg_contours) // 2)
    
    return egg_count

def detect_food_requirements(img):
    """Detect food requirements (icons in top left)"""
    height, width = img.shape[:2]
    
    # Extract the top-left region where food icons are located
    # Adjust the region to better capture the food area (15% width, 20% height)
    food_region = img[0:int(height*0.20), 0:int(width*0.15)]
    
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(food_region, cv2.COLOR_BGR2HSV)
    
    # Define color ranges for different food types
    # Adjusted green range for worms - more permissive to catch the green worm
    lower_green = np.array([35, 25, 25])
    upper_green = np.array([90, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Gray for rats (using saturation to detect gray)
    lower_gray = np.array([0, 0, 40])
    upper_gray = np.array([180, 30, 200])
    gray_mask = cv2.inRange(hsv, lower_gray, upper_gray)
    
    # Count food items based on contours
    worm_count = count_food_items(green_mask)
    rat_count = count_food_items(gray_mask)
    
    # If no worms detected but there's a significant amount of green in the region,
    # it's likely there's a worm that wasn't properly segmented
    if worm_count == 0:
        green_pixels = np.sum(green_mask > 0)
        if green_pixels > 100:  # Threshold for minimum green pixels
            worm_count = 1
    
    return {"worms": worm_count, "rats": rat_count}

def count_food_items(mask):
    """Count food items from a binary mask"""
    # Apply morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 30:  # Reduced minimum area threshold to catch smaller food icons
            valid_contours.append(contour)
    
    return len(valid_contours)

def extract_card_text(img):
    """Extract text from the card (name and ability text)"""
    height, width = img.shape[:2]
    
    # Extract the name region (top middle)
    name_region = img[0:int(height*0.15), int(width*0.15):int(width*0.85)]
    
    # Extract the ability text region (bottom brown section)
    ability_region = img[int(height*0.7):height, 0:width]
    
    # Process name region
    name_gray = cv2.cvtColor(name_region, cv2.COLOR_BGR2GRAY)
    _, name_thresh = cv2.threshold(name_gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Process ability region
    ability_gray = cv2.cvtColor(ability_region, cv2.COLOR_BGR2GRAY)
    _, ability_thresh = cv2.threshold(ability_gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Use OCR to extract text
    name_text = pytesseract.image_to_string(name_thresh).strip()
    ability_text = pytesseract.image_to_string(ability_thresh).strip()
    
    # Create a binary image with only the text
    text_only = np.zeros_like(img)
    text_only[0:int(height*0.15), int(width*0.15):int(width*0.85)] = cv2.cvtColor(name_thresh, cv2.COLOR_GRAY2BGR)
    text_only[int(height*0.7):height, 0:width] = cv2.cvtColor(ability_thresh, cv2.COLOR_GRAY2BGR)
    
    return {
        "name": name_text,
        "ability": ability_text,
        "text_only_image": text_only
    }

def analyze_wingspan_card(image_path):
    """Main function to analyze a Wingspan card"""
    # Load and preprocess the image
    img, gray = preprocess_image(image_path)
    
    # Detect card features
    victory_points = detect_victory_points(img)
    egg_capacity = detect_egg_capacity(img)
    food_requirements = detect_food_requirements(img)
    text_info = extract_card_text(img)
    
    # Compile results
    results = {
        "victory_points": victory_points,
        "egg_capacity": egg_capacity,
        "food_requirements": food_requirements,
        "name": text_info["name"],
        "ability": text_info["ability"],
        "text_only_image": text_info["text_only_image"]
    }
    
    return results

def display_results(results, original_img):
    """Display the analysis results"""
    # Create a copy of the original image for visualization
    vis_img = original_img.copy()
    height, width = vis_img.shape[:2]
    
    # Add text annotations
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(vis_img, f"VP: {results['victory_points']}", 
                (10, 30), font, 1, (0, 0, 255), 2)
    cv2.putText(vis_img, f"Eggs: {results['egg_capacity']}", 
                (10, 70), font, 1, (0, 0, 255), 2)
    
    food_text = f"Food: {results['food_requirements']['worms']} worms, {results['food_requirements']['rats']} rats"
    cv2.putText(vis_img, food_text, 
                (10, 110), font, 1, (0, 0, 255), 2)
    
    # Display the results
    show_image("Original Card", original_img)
    show_image("Analysis Results", vis_img)
    show_image("Text Only", results["text_only_image"])
    
    # Print text information
    print(f"Card Name: {results['name']}")
    print(f"Ability Text: {results['ability']}")
    print(f"Victory Points: {results['victory_points']}")
    print(f"Egg Capacity: {results['egg_capacity']}")
    print(f"Food Requirements: {results['food_requirements']}")

if __name__ == "__main__":
    # Test with the provided images
    image_paths = ["phone1.pdf", "phone2.pdf"]
    
    for path in image_paths:
        print(f"\nAnalyzing {path}...")
        try:
            img, _ = preprocess_image(path)
            results = analyze_wingspan_card(path)
            display_results(results, img)
        except Exception as e:
            print(f"Error processing {path}: {e}") 