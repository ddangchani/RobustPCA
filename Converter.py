# Convert Video file to Matrix Data for Robust PCA

import numpy as np
import cv2
import sys

def convert_video(path, gray=True):
    """
    Convert Video file to Matrix Data for Robust PCA
    """
    cap = cv2.VideoCapture(path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if gray:
        data = np.zeros((frame_count, frame_height, frame_width), dtype=np.uint8)

        for i in range(frame_count):
            ret, frame = cap.read()
            if ret:
                data[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                print("Error: Video is not read.")
                sys.exit(1)
    else:
        data = np.zeros((frame_count, frame_height, frame_width, 3), dtype=np.uint8)

        for i in range(frame_count):
            ret, frame = cap.read()
            if ret:
                data[i] = frame
            else:
                print("Error: Video is not read.")
                sys.exit(1)

    cap.release()
    print("Video is read successfully.")
    print("Shape : ", data.shape)
    return data