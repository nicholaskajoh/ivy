import numpy as np
import cv2


def get_roi_frame(current_frame, polygon):
    mask = np.zeros(current_frame.shape, dtype=np.uint8)
    polygon = np.array([polygon], dtype=np.int32)
    num_frame_channels = current_frame.shape[2]
    mask_ignore_color = (255,) * num_frame_channels
    cv2.fillPoly(mask, polygon, mask_ignore_color)
    masked_frame = cv2.bitwise_and(current_frame, mask)
    return masked_frame

def draw_roi(frame, polygon):
    frame_overlay = frame.copy()
    polygon = np.array([polygon], dtype=np.int32)
    cv2.fillPoly(frame_overlay, polygon, (0, 255, 255))
    alpha = 0.3
    output_frame = cv2.addWeighted(frame_overlay, alpha, frame, 1 - alpha, 0)
    return output_frame