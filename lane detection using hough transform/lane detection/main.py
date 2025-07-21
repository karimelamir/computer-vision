import cv2
from collections import deque
from hugh_transform import *  # Make sure hough_transform and lane_lines are defined
import numpy as np

QUEUE_LENGTH = 128

class LaneDetector:
    def __init__(self, angle_threshold=60):
        self.left_lines = deque(maxlen=QUEUE_LENGTH)
        self.right_lines = deque(maxlen=QUEUE_LENGTH)
        self.angle_threshold = angle_threshold  # Adjustable threshold for lane departure detection

    def process(self, origin_image, preprocessed_image):
        # Detect lines in the preprocessed image
        lines = hough_lines(preprocessed_image)
        left_line, right_line = lane_lines(origin_image, lines)

        # Average the lines over recent frames for stability
        left_line = self._mean_line(left_line, self.left_lines)
        right_line = self._mean_line(right_line, self.right_lines)

        # Draw warning if lane departure is detected
        self._lane_departure_warning(origin_image, left_line, right_line)

        # Draw and return the lane lines on the original image
        return draw_lane_lines(origin_image, (left_line, right_line), thickness=10)

    def _mean_line(self, line, lines):
        """Averages lane lines over recent frames for smooth detection."""
        if line is not None:
            lines.append(line)
        if len(lines) > 0:
            line = np.mean(lines, axis=0, dtype=np.int32)
            line = tuple(map(tuple, line))  # Convert numpy array to tuples
        return line

    def _calculate_angle(self, p1, p2):
        """Calculates the angle of a line segment defined by two points."""
        dx, dy = p2[0] - p1[0], p1[1] - p2[1]
        return np.arctan2(dy, dx) * (180 / np.pi)

    def _lane_departure_warning(self, origin_image, left_line, right_line):
        """Displays a lane departure warning if angle thresholds are exceeded."""
        font = cv2.FONT_HERSHEY_SIMPLEX

        if left_line is not None and right_line is not None:
            (xl1, yl1), (xl2, yl2) = left_line
            (xr1, yr1), (xr2, yr2) = right_line

            if abs(xr2 - xl2) < 50:
                draw_lane_lines(origin_image, (None, None), color=[0, 0, 0], thickness=0)
                return

            left_angle = self._calculate_angle((xl1, yl1), (xl2, yl2))
            right_angle = self._calculate_angle((xr1, yr1), (xr2, yr2))

            if left_angle > self.angle_threshold:
                cv2.putText(origin_image, "Left", (10, 30), font, 0.6, (0, 0, 255), 2)
            elif right_angle > self.angle_threshold:
                cv2.putText(origin_image, "Right", (origin_image.shape[1] - 60, 30), font, 0.6, (0, 0, 255), 2)

        elif left_line is None and right_line is not None:
            (xr1, yr1), (xr2, yr2) = right_line
            right_angle = self._calculate_angle((xr1, yr1), (xr2, yr2))
            if right_angle > self.angle_threshold:
                cv2.putText(origin_image, "Right", (origin_image.shape[1] - 60, 30), font, 0.6, (0, 0, 255), 2)

        elif right_line is None and left_line is not None:
            (xl1, yl1), (xl2, yl2) = left_line
            left_angle = self._calculate_angle((xl1, yl1), (xl2, yl2))
            if xl2 > origin_image.shape[1] / 2:
                draw_lane_lines(origin_image, (None, None), color=[0, 0, 0], thickness=0)
                return
            if left_angle > self.angle_threshold:
                cv2.putText(origin_image, "Left", (10, 30), font, 0.6, (0, 0, 255), 2)

def main():
    # Initialize the LaneDetector
    lane_detector = LaneDetector()

    # Open the default camera (usually the laptop webcam)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame (apply any necessary preprocessing like grayscale, edge detection)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        preprocessed_frame = cv2.Canny(gray, 50, 150)  # Example of edge detection

        # Process the frame to detect lanes
        output_frame = lane_detector.process(frame, preprocessed_frame)

        # Display the resulting frame
        cv2.imshow("Lane Detection", output_frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
