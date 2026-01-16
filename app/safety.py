import cv2
import numpy as np

def is_blurry(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score < 100

def validate_image(image_np):
    """
    Returns True if image is good, else False with message.
    """
    if is_blurry(image_np):
        return False, "The image appears blurry. Please upload a clearer image."
    return True, None

