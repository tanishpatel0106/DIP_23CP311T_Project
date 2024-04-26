import cv2
import numpy as np

def calculate_density(binary_image):
    return np.sum(binary_image) / (binary_image.size * 255)

def calculate_transitions(binary_image):
    horizontal_transitions = np.sum(np.abs(np.diff(binary_image, axis=1))) / 255
    vertical_transitions = np.sum(np.abs(np.diff(binary_image, axis=0))) / 255
    return horizontal_transitions, vertical_transitions

def compute_variation(contours):
    heights = [cv2.boundingRect(contour)[3] for contour in contours]
    widths = [cv2.boundingRect(contour)[2] for contour in contours]
    return np.std(heights), np.std(widths) if heights and widths else (0, 0)

def compute_components(binary_image):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image)
    sizes = stats[1:, cv2.CC_STAT_AREA]
    return num_labels - 1, np.max(sizes) if sizes.size > 0 else 0

def compute_run_lengths(binary_image):
    max_run_length = 0
    total_run_length = 0
    run_count = 0
    for row in binary_image:
        runs = np.where(row[:-1] != row[1:])[0]  # transition points
        run_lengths = np.diff(runs)  # lengths of runs
        if run_lengths.size > 0:
            max_run_length = max(max_run_length, np.max(run_lengths))
            total_run_length += np.sum(run_lengths)
            run_count += run_lengths.size
    avg_run_length = total_run_length / run_count if run_count > 0 else 0
    return max_run_length, avg_run_length

def calculate_distance_between_hulls(hull1, hull2):
    hull1_points = hull1.reshape(-1, 2)
    hull2_points = hull2.reshape(-1, 2)
    min_dist = np.min([np.linalg.norm(p1 - p2) for p1 in hull1_points for p2 in hull2_points])
    return min_dist

def determine_relative_location(features1, features2):
    if features1['centroid_x'] < features2['centroid_x']:
        return 'left'
    elif features1['centroid_x'] > features2['centroid_x']:
        return 'right'
    elif features1['centroid_y'] < features2['centroid_y']:
        return 'top'
    else:
        return 'bottom'

def extract_state_features(contour, binary_image):
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h
    hull = cv2.convexHull(contour)
    num_points_in_convex_hull = len(hull)

    patch = binary_image[y:y+h, x:x+w]
    density = calculate_density(patch)
    horizontal_transitions, vertical_transitions = calculate_transitions(patch)
    
    return {
        'x': x,
        'y': y,
        'width': w,
        'height': h,
        'aspect_ratio': aspect_ratio,
        'density': density,
        'horizontal_transitions': horizontal_transitions,
        'vertical_transitions': vertical_transitions,
        'num_points_in_convex_hull': num_points_in_convex_hull,
        'centroid_x': x + w / 2,
        'centroid_y': y + h / 2,
    }

def extract_features_from_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Extract global state features
    num_components, max_component_size = compute_components(binary_image)
    max_run_length, avg_run_length = compute_run_lengths(binary_image)

    # Find contours/components
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])  # Sort contours left-to-right

    # Extract state features for each contour
    state_features = [extract_state_features(c, binary_image) for c in contours]

    # Compute transitional features
    transition_features_list = []
    for i in range(len(contours) - 1):
        hull1 = cv2.convexHull(contours[i])
        hull2 = cv2.convexHull(contours[i + 1])
        
        features = {}
        features['relative_location'] = determine_relative_location(state_features[i], state_features[i + 1])
        features['convex_hull_distance'] = calculate_distance_between_hulls(hull1, hull2)
        features['ratio_of_aspect_ratios'] = state_features[i]['aspect_ratio'] / state_features[i + 1]['aspect_ratio']
        features['ratio_of_number_of_components'] = num_components  # This would be the same for all as we consider the whole image

        transition_features_list.append(features)

    # Combine all features
    all_features = {
        'global': {
            'num_components': num_components,
            'max_component_size': max_component_size,
            'max_run_length': max_run_length,
            'avg_run_length': avg_run_length,
        },
        'state': state_features,
        'transition': transition_features_list,
    }
    return all_features

# Usage
image_path = '/home/tanishpatel01/DIP_23CP311T_Project/Test_Image.jpeg'
all_features = extract_features_from_image(image_path)
print(all_features)
