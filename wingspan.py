import cv2
import pytesseract
import numpy as np
import os
import re
import sys
import json
from datetime import datetime
from collections import Counter

def ensure_output_dir():
    """Create the output directory if it doesn't exist"""
    output_dir = "generated_images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def ensure_individual_cards_dir():
    """Create the individual cards directory if it doesn't exist"""
    output_dir = "wingspan_individual_cards"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def get_output_path(image_path, suffix):
    """Generate a path in the output directory with the given suffix"""
    output_dir = ensure_output_dir()
    base_name = os.path.basename(image_path)
    name_without_ext = os.path.splitext(base_name)[0]
    return os.path.join(output_dir, f"{name_without_ext}_{suffix}.jpg")

def check_template_files():
    """Check if all required template files exist"""
    templates = [
        "egg_template.jpg",
        "wheat_template.jpg",
        "worm_template.jpg",
        "fruit_template.jpg",
        "fish_template.jpg",
        "wild_template.jpg",
        "rat_template.jpg",
        "wetland_template.jpg",
        "prairie_template.jpg",
        "forest_template.jpg"
    ]
    
    missing_templates = []
    for template in templates:
        if not os.path.exists(template):
            missing_templates.append(template)
    
    if missing_templates:
        print(f"WARNING: The following template files are missing: {', '.join(missing_templates)}")
        print("Template detection may not work correctly.")
    
    return len(missing_templates) == 0

def split_grid_image(grid_image_path):
    """
    Split a grid image of cards (6 wide x 4 high) into individual card images
    Each card is expected to be approximately 672px wide by 1018px high
    
    Args:
        grid_image_path: Path to the grid image
        
    Returns:
        List of paths to individual card images
    """
    # Read the grid image
    grid_image = cv2.imread(grid_image_path)
    if grid_image is None:
        raise ValueError(f"Could not read image at {grid_image_path}")
    
    # Get the dimensions of the grid image
    height, width = grid_image.shape[:2]
    
    # Define grid dimensions
    grid_width = 6
    grid_height = 4
    
    # Calculate card dimensions based on the image size
    card_width = width // grid_width
    card_height = height // grid_height
    
    print(f"Grid image dimensions: {width}x{height}")
    print(f"Calculated card dimensions: {card_width}x{card_height}")
    
    # Create output directories
    output_dir = ensure_output_dir()
    individual_cards_dir = ensure_individual_cards_dir()
    base_name = os.path.basename(grid_image_path)
    name_without_ext = os.path.splitext(base_name)[0]
    
    # List to store paths of extracted card images
    card_image_paths = []
    
    # Extract each card from the grid
    for row in range(grid_height):
        for col in range(grid_width):
            # Calculate coordinates for this card
            x = col * card_width
            y = row * card_height
            
            # Make sure we're not exceeding the image dimensions
            if x + card_width > width or y + card_height > height:
                print(f"Warning: Card at position ({row}, {col}) exceeds image boundaries")
                continue
            
            # Extract the card image
            card_image = grid_image[y:y+card_height, x:x+card_width]
            
            # Generate a temporary path for this card to extract the bird name
            temp_card_path = os.path.join(output_dir, f"{name_without_ext}_card_r{row}_c{col}.jpg")
            cv2.imwrite(temp_card_path, card_image)
            
            try:
                # Extract the bird name
                bird_name, _, _ = extract_bird_name(temp_card_path)
                # Skip if bird name is empty or just whitespace
                if not bird_name or bird_name.isspace():
                    print(f"Skipping card at position ({row}, {col}) - no bird name detected")
                    continue
                    
                # Replace spaces with underscores and remove any invalid characters
                safe_bird_name = re.sub(r'[^a-zA-Z0-9_]', '', bird_name.replace(' ', '_'))
                # Generate the final path with the bird name
                final_card_path = os.path.join(individual_cards_dir, f"{safe_bird_name}.jpg")
                # Save the card image with the bird name
                cv2.imwrite(final_card_path, card_image)
                print(f"Saved card as: {safe_bird_name}.jpg")
            except Exception as e:
                print(f"Error processing card at position ({row}, {col}): {str(e)}")
                continue
            
            # Add the path to our list
            card_image_paths.append(temp_card_path)
            
            print(f"Extracted card at position ({row}, {col})")
    
    print(f"Extracted {len(card_image_paths)} cards from grid image")
    return card_image_paths

def extract_text(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    text = pytesseract.image_to_string(thresh, config='--psm 11')
    return text

def detect_victory_points(image_path):
    image = cv2.imread(image_path)
    
    # Define a more precise region of interest for victory points
    height, width = image.shape[:2]
    
    # Calculate ROI coordinates (left side, about 1/3 down from the top)
    x_start = int(width * 0.05)
    x_end = int(width * 0.15)
    y_start = int(height * 0.3)
    y_end = int(height * 0.45)
    
    # Extract the region of interest
    roi = image[y_start:y_end, x_start:x_end]
    
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Apply binary inverse threshold (the most effective method)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    
    # Dictionary to track which method produced which result
    detection_methods = {}
    
    # Try to recognize text with different configurations
    custom_config = r'--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789'
    
    # Try all methods with the binary threshold image
    vp_candidates = []
    
    # Method 1: Standard OCR
    text = pytesseract.image_to_string(thresh, config=custom_config).strip()
    if text and text.isdigit():
        vp_candidates.append(text)
        detection_methods[text] = "Standard OCR with Binary Inverse threshold"
    
    # Method 2: Try with different PSM modes
    for psm in [6, 7, 8, 9, 10, 13]:
        config = f'--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789'
        text = pytesseract.image_to_string(thresh, config=config).strip()
        if text and text.isdigit():
            vp_candidates.append(text)
            detection_methods[text] = f"PSM mode {psm} on Binary Inverse threshold"
    
    # Method 3: Try with image scaling
    scaled_roi = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    text = pytesseract.image_to_string(scaled_roi, config=custom_config).strip()
    if text and text.isdigit():
        vp_candidates.append(text)
        detection_methods[text] = "2x scaled Binary Inverse threshold"
    
    # Method 4: Try contour detection and analysis
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Sort contours by area and get the largest one (likely to be the number)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        # Create a mask for the largest contour
        mask = np.zeros_like(thresh)
        cv2.drawContours(mask, [contours[0]], -1, 255, -1)
        # Apply the mask
        result = cv2.bitwise_and(thresh, mask)
        text = pytesseract.image_to_string(result, config=custom_config).strip()
        if text and text.isdigit():
            vp_candidates.append(text)
            detection_methods[text] = "Contour analysis on Binary Inverse threshold"
    
    # Choose the most common result or the first valid one
    vp = ''
    successful_method = "None"
    if vp_candidates:
        counter = Counter(vp_candidates)
        most_common = counter.most_common(1)[0][0]
        vp = most_common
        successful_method = detection_methods.get(vp, "Unknown method")
    
    # If still no valid result, try a hardcoded approach for common values
    if not vp:
        # Check if it's likely a "5" based on pixel analysis
        white_pixels = np.sum(thresh == 255)
        total_pixels = thresh.size
        white_ratio = white_pixels / total_pixels
        
        # Typical white pixel ratios for different numbers (these values need tuning)
        if 0.15 <= white_ratio <= 0.25:  # Approximate range for "5"
            vp = "5"
            successful_method = f"Pixel ratio analysis ({white_ratio:.3f})"
        elif white_ratio < 0.15:  # Smaller numbers like "1"
            vp = "1"
            successful_method = f"Pixel ratio analysis ({white_ratio:.3f})"
        elif white_ratio > 0.25:  # Larger numbers
            vp = "8"  # Default to 8 for larger patterns
            successful_method = f"Pixel ratio analysis ({white_ratio:.3f})"
    
    # Draw a rectangle around the detected area on the original image
    cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (255, 0, 255), 2)
    cv2.putText(image, f"VP: {vp}", (x_start, y_start - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    
    # Add method information to the image
    method_text = f"Method: {successful_method}"
    cv2.putText(image, method_text, (x_start, y_end + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    
    # Save the annotated image
    output_path = get_output_path(image_path, "vp_detected")
    cv2.imwrite(output_path, image)
    
    # Create a debug info dictionary
    debug_info = {
        'value': vp,
        'method': successful_method,
        'candidates': dict(Counter(vp_candidates)),
        'white_pixel_ratio': white_pixels / total_pixels if 'total_pixels' in locals() else None,
    }
    
    return vp, output_path, debug_info

def detect_eggs(image_path):
    image = cv2.imread(image_path)
    egg_icon = cv2.imread('egg_template.jpg', 0)  # Template matching for eggs
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(gray, egg_icon, cv2.TM_CCOEFF_NORMED)
    threshold = 0.92
    locations = np.where(result >= threshold)
    locations = list(zip(*locations[::-1]))  # Convert to list of (x, y) positions
    
    # Get dimensions of the template
    w, h = egg_icon.shape[::-1]
    
    # Filter out overlapping detections
    filtered_locations = []
    if locations:
        # Convert locations to rectangles format for non-maximum suppression
        rectangles = []
        for loc in locations:
            rect = [int(loc[0]), int(loc[1]), int(loc[0] + w), int(loc[1] + h)]
            rectangles.append(rect)
        
        # Convert to numpy array for processing
        rectangles = np.array(rectangles)
        
        # Define a function for calculating overlap (IoU - Intersection over Union)
        def calculate_iou(box1, box2):
            # Calculate intersection area
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            
            if x2 < x1 or y2 < y1:
                return 0.0
                
            intersection_area = (x2 - x1) * (y2 - y1)
            
            # Calculate union area
            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union_area = box1_area + box2_area - intersection_area
            
            return intersection_area / union_area
        
        # Simple non-maximum suppression
        selected_indices = []
        for i in range(len(rectangles)):
            keep = True
            for j in selected_indices:
                if calculate_iou(rectangles[i], rectangles[j]) > 0.3:  # Threshold for overlap
                    keep = False
                    break
            if keep:
                selected_indices.append(i)
        
        # Get the filtered locations
        filtered_locations = [locations[i] for i in selected_indices]
    
    # Draw rectangles around each filtered match
    for loc in filtered_locations:
        top_left = loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 2)  # Red color for eggs
        # Add label
        cv2.putText(image, "egg", (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Save the image with bounding boxes
    output_path = get_output_path(image_path, "eggs_detected")
    cv2.imwrite(output_path, image)
    
    return len(filtered_locations), output_path

def detect_food(image_path):
    image = cv2.imread(image_path)
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Define the top-left region where food requirements are located
    roi_width = int(width * 0.40)
    roi_height = int(height * 0.22)
    roi = image[0:roi_height, 0:roi_width]
    
    # Save the ROI image for debugging
    roi_output_path = get_output_path(image_path, "food_roi")
    cv2.imwrite(roi_output_path, roi)
    
    # Load all food templates
    wheat_icon = cv2.imread('wheat_template.jpg', 0)
    worm_icon = cv2.imread('worm_template.jpg', 0)
    fruit_icon = cv2.imread('fruit_template.jpg', 0)
    fish_icon = cv2.imread('fish_template.jpg', 0)
    wild_icon = cv2.imread('wild_template.jpg', 0)
    rat_icon = cv2.imread('rat_template.jpg', 0)
    
    # Convert ROI to grayscale
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Dictionary to store food counts and their locations
    food_counts = {
        'wheat': 0,
        'worm': 0,
        'fruit': 0,
        'fish': 0,
        'wild': 0,
        'rat': 0
    }
    
    def detect_food_type(template, food_name, color):
        result = cv2.matchTemplate(gray_roi, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.82  # Slightly lower threshold to catch more potential matches
        locations = np.where(result >= threshold)
        locations = list(zip(*locations[::-1]))  # Convert to list of (x, y) positions
        
        # Get dimensions of the template
        w, h = template.shape[::-1]
        
        # Filter out overlapping detections
        filtered_locations = []
        if locations:
            # Convert locations to rectangles format for non-maximum suppression
            rectangles = []
            for loc in locations:
                rect = [int(loc[0]), int(loc[1]), int(loc[0] + w), int(loc[1] + h)]
                rectangles.append(rect)
            
            # Convert to numpy array for processing
            rectangles = np.array(rectangles)
            
            # Simple non-maximum suppression
            selected_indices = []
            for i in range(len(rectangles)):
                keep = True
                for j in selected_indices:
                    # Calculate IoU
                    box1 = rectangles[i]
                    box2 = rectangles[j]
                    
                    # Calculate intersection area
                    x1 = max(box1[0], box2[0])
                    y1 = max(box1[1], box2[1])
                    x2 = min(box1[2], box2[2])
                    y2 = min(box1[3], box2[3])
                    
                    if x2 < x1 or y2 < y1:
                        intersection_area = 0
                    else:
                        intersection_area = (x2 - x1) * (y2 - y1)
                    
                    # Calculate union area
                    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
                    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
                    union_area = box1_area + box2_area - intersection_area
                    
                    if intersection_area / union_area > 0.3:  # Threshold for overlap
                        keep = False
                        break
                if keep:
                    selected_indices.append(i)
            
            # Get the filtered locations
            filtered_locations = [locations[i] for i in selected_indices]
        
        # Draw rectangles around each filtered match (on the original image)
        for loc in filtered_locations:
            top_left = loc  # Location is already in ROI coordinates
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv2.rectangle(roi, top_left, bottom_right, color, 2)
            # Add label
            cv2.putText(roi, food_name, (top_left[0], top_left[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return len(filtered_locations)
    
    # Detect each food type with a different color
    food_counts['wheat'] = detect_food_type(wheat_icon, "wheat", (0, 255, 0))  # Green
    food_counts['worm'] = detect_food_type(worm_icon, "worm", (0, 255, 255))  # Yellow
    food_counts['fruit'] = detect_food_type(fruit_icon, "fruit", (0, 0, 255))  # Red
    food_counts['fish'] = detect_food_type(fish_icon, "fish", (255, 0, 0))  # Blue
    food_counts['wild'] = detect_food_type(wild_icon, "wild", (255, 0, 255))  # Magenta
    food_counts['rat'] = detect_food_type(rat_icon, "rat", (128, 128, 128))  # Gray
    
    # Save the image with bounding boxes
    output_path = get_output_path(image_path, "food_detected")
    cv2.imwrite(output_path, image)
    
    # Remove food types with zero count
    food_requirements = {k: v for k, v in food_counts.items() if v > 0}
    
    return food_requirements, output_path

def detect_habitats(image_path):
    image = cv2.imread(image_path)
    # Load habitat templates
    wetland_icon = cv2.imread('wetland_template.jpg', 0)
    prairie_icon = cv2.imread('prairie_template.jpg', 0)
    forest_icon = cv2.imread('forest_template.jpg', 0)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    habitats = {
        'wetland': False,
        'prairie': False,
        'forest': False
    }
    
    def detect_habitat(template, habitat_name, color):
        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.85  # Adjust threshold as needed
        locations = np.where(result >= threshold)
        locations = list(zip(*locations[::-1]))  # Convert to list of (x, y) positions
        
        # Get dimensions of the template
        w, h = template.shape[::-1]
        
        # Filter out overlapping detections
        filtered_locations = []
        if locations:
            # Convert locations to rectangles format for non-maximum suppression
            rectangles = []
            for loc in locations:
                rect = [int(loc[0]), int(loc[1]), int(loc[0] + w), int(loc[1] + h)]
                rectangles.append(rect)
            
            # Convert to numpy array for processing
            rectangles = np.array(rectangles)
            
            # Simple non-maximum suppression
            selected_indices = []
            for i in range(len(rectangles)):
                keep = True
                for j in selected_indices:
                    # Calculate IoU
                    box1 = rectangles[i]
                    box2 = rectangles[j]
                    
                    # Calculate intersection area
                    x1 = max(box1[0], box2[0])
                    y1 = max(box1[1], box2[1])
                    x2 = min(box1[2], box2[2])
                    y2 = min(box1[3], box2[3])
                    
                    if x2 < x1 or y2 < y1:
                        intersection_area = 0
                    else:
                        intersection_area = (x2 - x1) * (y2 - y1)
                    
                    # Calculate union area
                    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
                    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
                    union_area = box1_area + box2_area - intersection_area
                    
                    if intersection_area / union_area > 0.3:  # Threshold for overlap
                        keep = False
                        break
                if keep:
                    selected_indices.append(i)
            
            # Get the filtered locations
            filtered_locations = [locations[i] for i in selected_indices]
        
        # Draw rectangles around each filtered match
        for loc in filtered_locations:
            top_left = loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv2.rectangle(image, top_left, bottom_right, color, 2)
            # Add label
            cv2.putText(image, habitat_name, (top_left[0], top_left[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Return True if at least one match was found
        return len(filtered_locations) > 0
    
    # Detect each habitat with a different color
    habitats['wetland'] = detect_habitat(wetland_icon, "wetland", (255, 0, 0))  # Blue
    habitats['prairie'] = detect_habitat(prairie_icon, "prairie", (0, 165, 255))  # Orange
    habitats['forest'] = detect_habitat(forest_icon, "forest", (0, 128, 0))  # Green
    
    # Save the image with bounding boxes
    output_path = get_output_path(image_path, "habitats_detected")
    cv2.imwrite(output_path, image)
    
    return habitats, output_path

def extract_bird_name(image_path):
    """
    Extract the bird name from a card image by focusing on the name region
    
    Args:
        image_path: Path to the card image
        
    Returns:
        String containing the extracted bird name
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Define the region of interest for the bird name
    # The name appears in the top-right area of the card
    # Approximate coordinates (may need tuning for different card formats)
    x_start = int(width * 0.38)
    x_end = int(width * 0.95)
    y_start = int(height * 0.05)
    y_end = int(height * 0.10)
    
    # Extract the region containing the name
    name_roi = image[y_start:y_end, x_start:x_end]
    
    # Create an output path for debugging
    output_dir = ensure_output_dir()
    base_name = os.path.basename(image_path)
    name_without_ext = os.path.splitext(base_name)[0]
    roi_output_path = os.path.join(output_dir, f"{name_without_ext}_name_roi.jpg")
    cv2.imwrite(roi_output_path, name_roi)
    
    # Convert to grayscale
    gray = cv2.cvtColor(name_roi, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to isolate the dark text
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    
    # Apply some preprocessing to make the text clearer
    kernel = np.ones((1, 1), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    
    # Save the preprocessed image for debugging
    thresh_output_path = os.path.join(output_dir, f"{name_without_ext}_name_thresh.jpg")
    cv2.imwrite(thresh_output_path, thresh)
    
    # Use Tesseract with specific configuration for all caps text
    custom_config = '--psm 7 --oem 3'  # PSM 7 assumes a single line of text
    bird_name = pytesseract.image_to_string(thresh, config=custom_config).strip()
    
    # Clean up the extracted name
    # Remove any non-alphanumeric characters except spaces and apostrophes
    bird_name = re.sub(r'[^a-zA-Z0-9\s\']', '', bird_name)
    
    # Limit to first 3 words (most bird names are 1-3 words)
    words = bird_name.split()
    if len(words) > 3:
        bird_name = ' '.join(words[:3])
    
    # Convert to uppercase
    bird_name = bird_name.upper()
    
    return bird_name, roi_output_path, thresh_output_path

def process_card(image_path):
    """
    Process a single card image and detect its information
    
    Args:
        image_path: Path to the card image
        
    Returns:
        Dictionary with detected card information
    """
    # Ensure output directory exists
    ensure_output_dir()
    
    print(f"Processing card: {os.path.basename(image_path)}")
    
    try:
        bird_name, name_roi_path, name_thresh_path = extract_bird_name(image_path)
        print(f"  Detected bird name: {bird_name}")
    except Exception as e:
        print(f"  Error detecting bird name: {str(e)}")
        bird_name = "Unknown Bird"
        name_roi_path = None
        name_thresh_path = None
    
    try:
        victory_points, vp_output_image, vp_debug_info = detect_victory_points(image_path)
        print(f"  Detected victory points: {victory_points}")
    except Exception as e:
        print(f"  Error detecting victory points: {str(e)}")
        victory_points = "Error"
        vp_debug_info = {"error": str(e)}
        vp_output_image = None
    
    try:
        eggs, eggs_output_image = detect_eggs(image_path)
        print(f"  Detected eggs: {eggs}")
    except Exception as e:
        print(f"  Error detecting eggs: {str(e)}")
        eggs = 0
        eggs_output_image = None
    
    try:
        food_requirements, food_output_image = detect_food(image_path)
        print(f"  Detected food requirements: {food_requirements}")
    except Exception as e:
        print(f"  Error detecting food: {str(e)}")
        food_requirements = {}
        food_output_image = None
    
    try:
        habitats, habitats_output_image = detect_habitats(image_path)
        print(f"  Detected habitats: {habitats}")
    except Exception as e:
        print(f"  Error detecting habitats: {str(e)}")
        habitats = {"wetland": False, "prairie": False, "forest": False}
        habitats_output_image = None
    
    try:
        text = extract_text(image_path)
        # Only print the first 50 characters of text to keep output clean
        print(f"  Extracted text sample: {text[:50]}...")
    except Exception as e:
        print(f"  Error extracting text: {str(e)}")
        text = ""
    
    # Create a summary image with all detections if possible
    try:
        original = cv2.imread(image_path)
        summary_path = get_output_path(image_path, "summary")
        cv2.putText(original, f"Bird: {bird_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.putText(original, f"VP: {victory_points}, Eggs: {eggs}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(original, f"Food: {food_requirements}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(original, f"Habitats: {habitats}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.imwrite(summary_path, original)
    except Exception as e:
        print(f"  Error creating summary image: {str(e)}")
        summary_path = None
    
    print(f"  Completed processing card: {os.path.basename(image_path)}")
    
    return {
        'bird_name': bird_name,
        'victory_points': victory_points,
        'vp_debug_info': vp_debug_info,
        'egg_count': eggs,
        'food_requirements': food_requirements,
        'habitats': habitats,
        'extracted_text': text,
        'summary_image': summary_path
    }

def process_grid(grid_image_path):
    """
    Process a grid of cards (6 wide x 4 high) and detect information for each card
    
    Args:
        grid_image_path: Path to the grid image
        
    Returns:
        Dictionary mapping card positions to their processed information
    """
    # Split the grid image into individual card images
    print("Splitting grid image into individual cards...")
    card_image_paths = split_grid_image(grid_image_path)
    
    # Process each card and store the results
    results = {}
    print("\nProcessing individual cards:")
    print("----------------------------")
    
    for i, card_path in enumerate(card_image_paths):
        # Extract row and column from the filename
        filename = os.path.basename(card_path)
        
        # Progress indicator
        print(f"\nCard {i+1} of {len(card_image_paths)}: {filename}")
        
        try:
            # Extract row and column using regex for more reliable parsing
            match = re.search(r'card_r(\d+)_c(\d+)', filename)
            if match:
                row_idx = int(match.group(1))
                col_idx = int(match.group(2))
                
                # Process the card
                card_result = process_card(card_path)
                
                # Skip if bird name is empty or just whitespace
                if not card_result['bird_name'] or card_result['bird_name'].isspace():
                    print(f"Skipping card at position ({row_idx}, {col_idx}) - no bird name detected")
                    continue
                
                # Store the result with position information
                position_key = f"row_{row_idx}_col_{col_idx}"
                results[position_key] = {
                    'position': {'row': row_idx, 'column': col_idx},
                    'card_image_path': card_path,
                    'data': card_result
                }
                print(f"Successfully processed card at position ({row_idx}, {col_idx})")
            else:
                print(f"Could not extract row and column from filename: {filename}")
        except Exception as e:
            print(f"Error processing card {filename}: {str(e)}")
    
    print("\nCompleted processing all cards")
    return results

def save_results_to_json(results, image_path):
    """
    Save processing results to a JSON file
    
    Args:
        results: Dictionary of processing results
        image_path: Path to the original image
        
    Returns:
        Path to the JSON file
    """
    output_dir = ensure_output_dir()
    base_name = os.path.basename(image_path)
    name_without_ext = os.path.splitext(base_name)[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a serializable copy of the results
    serializable_results = {}
    for position, card_data in results.items():
        # Convert numpy values to native Python types where needed
        serializable_card = {
            'position': card_data['position'],
            'card_image_path': card_data['card_image_path'],
            'data': {}
        }
        
        data = card_data['data']
        serializable_card['data'] = {
            'bird_name': data.get('bird_name', "Unknown Bird"),
            'victory_points': data['victory_points'],
            'egg_count': int(data['egg_count']),
            'food_requirements': {k: int(v) for k, v in data['food_requirements'].items()},
            'habitats': {k: bool(v) for k, v in data['habitats'].items()},
            # Truncate text to avoid huge JSON files
            'extracted_text': data['extracted_text'][:500] if data['extracted_text'] else "",
            'summary_image': data['summary_image']
        }
        
        # Remove non-serializable objects from vp_debug_info
        if 'vp_debug_info' in data:
            debug_info = {}
            for k, v in data['vp_debug_info'].items():
                if isinstance(v, (dict, list, str, int, float, bool, type(None))):
                    debug_info[k] = v
                else:
                    debug_info[k] = str(v)
            serializable_card['data']['vp_debug_info'] = debug_info
        
        serializable_results[position] = serializable_card
    
    # Save to JSON file
    json_path = os.path.join(output_dir, f"{name_without_ext}_results_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Saved results to {json_path}")
    return json_path

if __name__ == "__main__":
    # Check if we're processing a single card or a grid
    if len(sys.argv) < 2:
        print("Usage: python wingspan.py <image_path> [--grid]")
        sys.exit(1)
    
    # Check for template files
    all_templates_exist = check_template_files()
    if not all_templates_exist:
        print("Continuing, but some detections may not work correctly.")
    
    image_path = sys.argv[1]
    is_grid = False
    
    if len(sys.argv) > 2 and sys.argv[2] == "--grid":
        is_grid = True
    
    try:
        if is_grid:
            print("\n=========================================")
            print("Processing grid of cards...")
            print("=========================================\n")
            result = process_grid(image_path)
            
            # Save results to JSON
            json_path = save_results_to_json(result, image_path)
            
            print("\n=========================================")
            print(f"SUMMARY: Processed {len(result)} cards from grid")
            print(f"Complete results saved to: {json_path}")
            print("=========================================\n")
            
            # Print a brief summary of each card
            for position, card_data in result.items():
                print(f"{position}:")
                print(f"  Victory Points: {card_data['data']['victory_points']}")
                print(f"  Eggs: {card_data['data']['egg_count']}")
                print(f"  Food: {card_data['data']['food_requirements']}")
                print(f"  Habitats: {card_data['data']['habitats']}")
                print("")
        else:
            print("Processing single card...")
            result = process_card(image_path)
            print("\nFinal Results:")
            print(f"Victory Points: {result['victory_points']}")
            print(f"Eggs: {result['egg_count']}")
            print(f"Food Requirements: {result['food_requirements']}")
            print(f"Habitats: {result['habitats']}")
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
