import os

# Path to the folder containing the label files
label_folder = "datasets\\test\\labels"

# Function to process each label file and strip it down to YOLO format
def strip_to_yolo_format(file_path):
    # Open the label file and read all lines
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # List to store corrected lines
    corrected_lines = []

    # Process each line in the label file
    for line in lines:
        # Split the line by spaces
        data = line.split()

        # Only take the first 5 values (class id, x_center, y_center, width, height)
        if len(data) >= 5:
            corrected_line = " ".join(data[:5]) + "\n"  # Join first 5 values and add newline
            corrected_lines.append(corrected_line)

    # Write the corrected lines back to the file
    with open(file_path, 'w') as f:
        f.writelines(corrected_lines)

# Iterate over all label files in the folder
for label_file in os.listdir(label_folder):
    if label_file.endswith(".txt"):  # Ensure it's a label file
        file_path = os.path.join(label_folder, label_file)
        strip_to_yolo_format(file_path)

print("All labels have been processed and stripped to YOLO format.")
