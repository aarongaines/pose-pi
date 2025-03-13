from typing import Tuple

import numpy as np
import onnxruntime as ort


def load_session(model_path) -> Tuple[ort.InferenceSession, dict]:
    session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
    session_info = {
        "input_name": session.get_inputs()[0].name,
        "input_shape": session.get_inputs()[0].shape,
        "input_type": session.get_inputs()[0].type,
        "output_names": [output.name for output in session.get_outputs()],
        "output_shape": session.get_outputs()[0].shape,
        "output_type": session.get_outputs()[0].type
    }
    return session, session_info


def get_pose(frame, session, session_info) -> Tuple:
    # Get the input and output names
    input_name = session_info["input_name"]
    output_names = session_info["output_names"]

    # Send the frame to the model
    input_data = np.transpose(frame, (2, 0, 1)) 
    input_data = np.expand_dims(input_data, axis=0)
    input_data = input_data.astype(np.float32)
    bboxes, keypoints = session.run(output_names, {input_name: input_data})

    return bboxes[0], keypoints[0]
