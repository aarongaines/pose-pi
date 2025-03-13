from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
import numpy as np
import cv2

from pose import rtmo

app = FastAPI()

model_path = "models/pose.onnx"
session, session_info = rtmo.load_session(model_path)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # For demonstration, assume the original frame is always 1920x1080,
    # and you always resize to 640x640 on the client.
    # If your client changes these dynamically, you must also send
    # the original (width,height) so the server knows how to scale back.

    ORIGINAL_W, ORIGINAL_H = 1920, 1080
    RESIZED_W, RESIZED_H = 640, 640
    THRESHOLD = 0.5

    # Precompute scale factors for going from 640→1920, 640→1080
    # (simple "squish" approach, ignoring aspect ratio differences).
    scale_x = ORIGINAL_W / float(RESIZED_W)  # e.g. 3.0 if 1920/640
    scale_y = ORIGINAL_H / float(RESIZED_H)  # e.g. 1.6875 if 1080/640

    try:
        while True:
            # 1) Receive raw binary from the client
            frame_bytes = await websocket.receive_bytes()

            # 2) Decode the JPEG bytes into a NumPy BGR image
            np_arr = np.frombuffer(frame_bytes, np.uint8)
            frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # 3) Convert to RGB if your model needs RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            bboxes, keypoints = rtmo.get_pose(frame_rgb, session, session_info)

            bboxes_original = []
            keypoints_original = []
            for i, bbox in enumerate(bboxes):
                if bbox[4] < THRESHOLD:
                    continue
                x1_orig = int(bbox[0] * scale_x)
                y1_orig = int(bbox[1] * scale_y)
                x2_orig = int(bbox[2] * scale_x)
                y2_orig = int(bbox[3] * scale_y)

                bboxes_original.append((x1_orig, y1_orig, x2_orig, y2_orig, bbox[4]))
            
                keypoints_for_bbox = []
                for j, (x_640, y_640, conf) in enumerate(keypoints[i]):
                    if conf < THRESHOLD:
                        continue
                    x_orig = int(x_640 * scale_x)
                    y_orig = int(y_640 * scale_y)
                    keypoints_for_bbox.append((x_orig, y_orig, conf))
                keypoints_original.append(keypoints_for_bbox)

            # Send the BBoxes and Keypoints back to the client
            # as a JSON array
            await websocket.send_json([bboxes_original, keypoints_original])

    except WebSocketDisconnect:
        print("Client disconnected")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)