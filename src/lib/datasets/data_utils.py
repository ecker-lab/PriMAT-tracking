import matplotlib.pylab as plt
import numpy as np
import cv2

color_dict = {
    0: (255, 0, 0),
    1: (255, 255, 0),
}


def plot_bbox(image, bboxes, labels, return_coords=False, colors=(0, 0, 225)):
    """Plot bounding boxes in image."""
    image = image.copy()
    coords = []
    for i in range(len(bboxes)):
        up_left = np.array([bboxes[i, 0], bboxes[i, 1]]).astype(int)
        low_right = np.array([bboxes[i, 2], bboxes[i, 3]]).astype(int)

        if isinstance(colors, dict):
            color = colors[labels[i]]
        else:
            color = colors

        image = cv2.rectangle(image, up_left, low_right, color=color, thickness=5)
        coords.append(up_left)

    if return_coords:
        return image, coords
    return image