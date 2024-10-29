import cv2
import json
import os
reference_points = []
def get_reference_points(path, frame_width, frame_height):
    # Mouse callback function to capture click events
    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            reference_points.append((x, y))
            print(f"Point captured: ({x}, {y})")
            cv2.circle(frame1, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Court", frame1)

    # Function to save reference points to a file
    def save_reference_points(file_path):
        with open(file_path, "w") as f:
            json.dump(reference_points, f)
        print(f"reference points saved to {file_path}")

    # Function to load reference points from a file
    def load_reference_points(file_path):
        global reference_points
        with open(file_path, "r") as f:
            reference_points = json.load(f)
        print(f"reference points loaded from {file_path}")

    # Load the frame (replace 'path_to_frame' with the actual path)
    if os.path.isfile("reference_points.json"):
        load_reference_points("reference_points.json")
        print(f"Loaded reference points: {reference_points}")
        return reference_points
    else:
        print(
            "No reference points file found. Please click on the court to set reference points."
        )
        cap2 = cv2.VideoCapture(path)
        if not cap2.isOpened():
            print("Error opening video file")
            exit()
        ret1, frame1 = cap2.read()
        if not ret1:
            print("Error reading video file")
            exit()
        frame1 = cv2.resize(frame1, (frame_width, frame_height))
        cv2.imshow("Court", frame1)
        cv2.setMouseCallback("Court", click_event)

        print(
            "Click on the key points of the court. Press 's' to save and 'q' to quit.\nMake sure to click in the following order shown by the example"
        )
        example_image = cv2.imread("output/annotated-squash-court.png")
        example_image_resized = cv2.resize(example_image, (frame_width, frame_height))
        cv2.imshow("Court Example", example_image_resized)
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                save_reference_points("reference_points.json")
                cv2.destroyAllWindows()
                return reference_points
            elif key == ord("q"):
                cv2.destroyAllWindows()
                return reference_points