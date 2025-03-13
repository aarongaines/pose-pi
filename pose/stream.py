import cv2
import numpy as np

import rtmo
import draw

def resize_and_pad(image, target_size=(416, 416)): # Ignore for now
    h, w, _ = image.shape
    scale = min(target_size[0] / h, target_size[1] / w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized_image = cv2.resize(image, (new_w, new_h))

    padded_image = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    pad_w = (target_size[1] - new_w) // 2
    pad_h = (target_size[0] - new_h) // 2

    padded_image[pad_h:pad_h+new_h, pad_w:pad_w+new_w, :] = resized_image

    return padded_image


def stream_webcam() -> None:
    # Load the pose model
    model_path = "../models/rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211/end2end.onnx"
    session, session_info = rtmo.load_session(model_path)

    cap = cv2.VideoCapture(0)  # or the index of your camera
    if not cap.isOpened():
        print("Could not open webcam.")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera.")
            break
        original_frame = frame.copy()
        # Resize to 640x640 for minimal bandwidth
        frame_640 = cv2.resize(frame, (640, 640))
        # get scale for x,y
        scale_x = original_frame.shape[1] / 640
        scale_y = original_frame.shape[0] / 640
        # Get pose estimation
        bboxes, keypoints = rtmo.get_pose(frame_640, session, session_info)
        # Adjust bounding boxes and keypoints to original frame size
        for bbox in bboxes:
            bbox[0] = int(bbox[0] * scale_x)
            bbox[1] = int(bbox[1] * scale_y)
            bbox[2] = int(bbox[2] * scale_x)
            bbox[3] = int(bbox[3] * scale_y)
        for i in range(len(keypoints)):
            for j in range(len(keypoints[i])):
                keypoints[i][j][0] = int(keypoints[i][j][0] * scale_x)
                keypoints[i][j][1] = int(keypoints[i][j][1] * scale_y)
        # Draw the bounding boxes and keypoints on the original frame
        frame = draw.draw_bboxes_keypoints(original_frame, bboxes, keypoints)
        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
        

if __name__ == "__main__":
    stream_webcam()