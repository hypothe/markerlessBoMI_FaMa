import math
import cv2

# Eye indices for blink detection
# Left eyes indices
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]

# right eyes indices
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]

# Colors
WHITE = (255, 255, 255)


def euclidean_distance(point_a, point_b):
    # Euclidean distance function

    x_a, y_a = point_a
    x_b, y_b = point_b
    distance = math.sqrt((x_a - x_b)**2 + (y_a - y_b)**2)

    return distance


def face_landmarks_detection(img, results, draw=False):
    img_height, img_width = img.shape[:2]
    # list[(x,y), (x,y)....]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw :
        [cv2.circle(img, p, 2, cv2.utils.GREEN, -1) for p in mesh_coord]

    # returning the list of tuples for each landmarks
    return mesh_coord


def blink_ratio(img, facelandmarks, right_indices, left_indices):
    # Blinking ratio

    # Right eyes
        # horizontal line
    rh_right = facelandmarks[right_indices[0]]
    rh_left = facelandmarks[right_indices[8]]
        # vertical line
    rv_top = facelandmarks[right_indices[12]]
    rv_bottom = facelandmarks[right_indices[4]]

    # Left eyes
        # horizontal line
    lh_right = facelandmarks[left_indices[0]]
    lh_left = facelandmarks[left_indices[8]]
        # vertical line
    lv_top = facelandmarks[left_indices[12]]
    lv_bottom = facelandmarks[left_indices[4]]

    # draw lines on right eye
    cv2.line(img, rh_right, rh_left, WHITE, 2)
    cv2.line(img, rv_top, rv_bottom, WHITE, 2)
    # draw lines on left eye
    cv2.line(img, lh_right, lh_left, WHITE, 2)
    cv2.line(img, lv_top, lv_bottom, WHITE, 2)

    rhDistance = euclidean_distance(rh_right, rh_left)
    rvDistance = euclidean_distance(rv_top, rv_bottom)

    lvDistance = euclidean_distance(lv_top, lv_bottom)
    lhDistance = euclidean_distance(lh_right, lh_left)

    if rvDistance != 0.0:
        reRatio = rhDistance/rvDistance
    else:
        reRatio = 1.0

    if lvDistance != 0.0:
        leRatio = lhDistance/lvDistance
    else:
        leRatio = 1.0

    return reRatio, leRatio
