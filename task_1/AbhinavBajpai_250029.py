import cv2
import numpy as np

def process_face(image_path, start_x, start_y, stride):

    image = cv2.imread(image_path)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    face_array = np.zeros((3, 3, 3))

    for row in range(3):
        for col in range(3):

            target_x = start_x + (col * stride)
            target_y = start_y + (row * stride)

            patch = hsv_image[
                target_y - 5 : target_y + 5,
                target_x - 5 : target_x + 5
            ]

            avg_hsv = np.mean(patch, axis=(0, 1))

            face_array[row][col] = avg_hsv

    return face_array


cube_data = np.zeros((6, 3, 3, 3))

faces = [
    "up.jpg",
    "down.jpg",
    "front.jpg",
    "back.jpg",
    "left.jpg",
    "right.jpg"
]

start_x = 100
start_y = 100
stride = 80

for i in range(6):
    cube_data[i] = process_face(
        faces[i],
        start_x,
        start_y,
        stride
    )

print(cube_data[0][0][0])