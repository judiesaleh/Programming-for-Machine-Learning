import os

def parse_ccpd_annotation(filename):
    """
    Parse CCPD annotation from the filename.
    Format: <unique_id>-<bbox1>-<bbox2>-<bbox3>-<bbox4>-<extra_info>.jpg
    Example: 305295138888888889-89_95-157&453_557&586-557&575_175&586_157&456_547&453-0_0_3_24_30_33_30_30-98-355.jpg
    """
    parts = filename.split('-')
    if len(parts) < 6:
        print(f"Skipping file {filename}: Invalid format")
        return None

    # Extract bounding box coordinates (from the 2nd and 3rd parts)
    bbox1 = parts[1].split('_')  # e.g., 89_95
    bbox2 = parts[2].split('&')  # e.g., 157&453_557&586
    bbox3 = parts[3].split('&')  # e.g., 557&575_175&586
    bbox4 = parts[4].split('&')  # e.g., 157&456_547&453

    try:
        # Extract (x_min, y_min) and (x_max, y_max)
        x_min = int(bbox2[0])
        y_min = int(bbox2[1].split('_')[0])
        x_max = int(bbox3[0])
        y_max = int(bbox3[1].split('_')[0])

        # Convert to YOLO format (x_center, y_center, width, height)
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min

        # Normalize to [0, 1]
        x_center /= 1000  # Assuming image width is 1000 (adjust if necessary)
        y_center /= 1000  # Assuming image height is 1000 (adjust if necessary)
        width /= 1000
        height /= 1000

        return [x_center, y_center, width, height]
    except (IndexError, ValueError) as e:
        print(f"Skipping file {filename}: Error parsing bounding box - {e}")
        return None

def convert_ccpd_to_yolo(dataset_path, output_path):
    """
    Convert CCPD dataset to YOLO format.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    images_dir = os.path.join(dataset_path, 'test/images') # change to right subset address for conversion to yolo
    labels_dir = os.path.join(output_path, 'labels')  # change to right subset address for conversion to yolo

    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)

    # Debug: Print dataset and output paths
    print(f"Dataset path: {dataset_path}")
    print(f"Output path: {output_path}")
    print(f"Images directory: {images_dir}")
    print(f"Labels directory: {labels_dir}")

    for filename in os.listdir(images_dir):
        if filename.endswith('.jpg'):
            image_path = os.path.join(images_dir, filename)
            label_path = os.path.join(labels_dir, filename.replace('.jpg', '.txt'))

            # Parse annotation
            bbox = parse_ccpd_annotation(filename)
            if bbox is None:
                continue

            # Write YOLO annotation
            with open(label_path, 'w') as f:
                f.write(f'0 {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n')
            print(f"Processed: {filename} -> {label_path}")

# Paths
dataset_path = r"C:\Users\judie\Desktop\Model\ccpd_green" # Path to the CCPD dataset folder
output_path = r"C:\Users\judie\Desktop\Model\ccpd_green"  # Output path for YOLO format dataset

# Convert CCPD to YOLO format
convert_ccpd_to_yolo(dataset_path, output_path)
print(f"Conversion complete. YOLO format dataset saved at: {output_path}")