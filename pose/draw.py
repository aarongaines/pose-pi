from dataclasses import dataclass

import numpy as np
import cv2


# Define the edges and colors for the keypoints
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]

KEYPOINT_COLORS = {
    "legs" : (255, 0, 0),
    "arms" : (0, 255, 0),
    "head" : (0, 0, 255),
    "torso" : (255, 255, 0),
}

@dataclass
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float

    def __str__(self):
        return f"({self.x1}, {self.y1}, {self.x2}, {self.y2}, {self.conf})"
    
    def from_array(array):
        """
        Create a BBox object from an array. The array should have 5 elements.
        The first four elements are the coordinates of the bounding box (x1, y1, x2, y2)
        and should be integers. The last element is the confidence score and should be a float.
        """
        return BBox(int(array[0]), int(array[1]), int(array[2]),int(array[3]), array[4])
    
    def to_array(self):
        return np.array([self.x1, self.y1, self.x2, self.y2, self.conf], dtype=np.float32)


def get_keypoint_color(index):
    if index in [5, 6, 7, 8, 9, 10]:
        return KEYPOINT_COLORS["arms"]
    elif index in [11, 12, 13, 14, 15, 16]:
        return KEYPOINT_COLORS["legs"]
    elif index in [0, 1, 2, 3, 4]:
        return KEYPOINT_COLORS["head"]
    else:
        return KEYPOINT_COLORS["torso"]
    

# Draw the bounding boxes, keypoints, and edges
def draw_bboxes_keypoints(image, bboxes, keypoints, threshold=0.5):
    image = image.copy()
    
    for i, bbox in enumerate(bboxes):
        bbox = BBox.from_array(bbox)

        cv2.rectangle(image, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), (0, 255, 0), 1)
        # Draw the bounding box index
        cv2.putText(image, str(i), (bbox.x1, bbox.y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        # Draw keypoints
        for j, (x, y, conf) in enumerate(keypoints[i]):
            if conf > threshold:  # Only draw keypoints with confidence > threshold
                color = get_keypoint_color(j)
                cv2.circle(image, (int(x), int(y)), 3, color, -1)
        
        # Draw edges
        for edge in COCO_EDGES:
            pt1 = keypoints[i][edge[0]]
            pt2 = keypoints[i][edge[1]]
            if pt1[2] > threshold and pt2[2] > threshold:  # Only draw edges if both keypoints have confidence > threshold
                color = get_keypoint_color(edge[0])
                cv2.line(image, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), color, 1)

    return image