import asyncio
import cv2
import websockets
import numpy as np

import draw

# Adjust to your server IP/port
SERVER_URI = "ws://192.168.1.145:5000/ws"

# Suppose your kiosk is also 1920x1080 (the original camera size).
# The server is expecting 640x640 images for inference.
RESIZED_W, RESIZED_H = 640, 640

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

def list_available_cameras(max_index=10):
    available_cameras = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            available_cameras.append(index)
            cap.release()
    return available_cameras




async def send_frames():
    cap = cv2.VideoCapture(0)  # or the index of your camera
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    async with websockets.connect(SERVER_URI) as websocket:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera.")
                break

            original_frame = frame.copy()  # keep a copy at 1920x1080

            # 1) Resize to 640x640 (simple squish) for minimal bandwidth
            frame_640 = cv2.resize(frame, (RESIZED_W, RESIZED_H))

            # 2) Encode as JPEG in-memory
            success, encoded = cv2.imencode(".jpg", frame_640)
            if not success:
                print("Failed to encode frame.")
                continue

            # 3) Send raw bytes over WebSocket
            await websocket.send(encoded.tobytes())

            # 4) Receive keypoints from server (JSON array)
            #    e.g.: [ [x1, y1, conf1], [x2, y2, conf2], ... ]
            data = await websocket.recv()
            # optional: import json; keypoints = json.loads(data)
            # but websockets with send_json might already decode it depending on usage
            # Usually you'd do:
            # keypoints = json.loads(data)
            import json
            bboxes, keypoints = json.loads(data)

            frame = draw.draw_bboxes_keypoints(frame, bboxes, keypoints)

            # 6) Display locally
            cv2.imshow("Pose Estimation", original_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    cameras = list_available_cameras()
    if cameras:
        print(f"Available cameras: {cameras}")
        asyncio.run(send_frames())
    else:
        print("No cameras found.")
    