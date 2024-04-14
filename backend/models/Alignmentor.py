import numpy as np
import os
from ultralytics import YOLO
from PIL import Image
import cv2


class Alignmentor:
    def __init__(self, path_to_weights, data_dir_path, weights_name):
        self.data_dir_path = data_dir_path

        self.model = YOLO(os.path.join(path_to_weights, weights_name))

    def align(self, img_file):
        print("Alingment...")

        result = self.model([img_file], task="detection")[0]  # return a list of Results objects
        boxes = result.boxes
        if len(boxes) >= 2:
            boxes = boxes[boxes.conf > 0.85]
        if boxes.xyxy.numel() == 0:
            return Image.open(img_file)

        x1, y1, x2, y2 = union_boxes(boxes.xyxy)

        im = Image.open(img_file)
        # left, upper, right, and lower
        # y1 -> y2
        im1 = im.crop((x1, y1, x2, y2))

        pil_image = im1.convert('RGB')
        open_cv_image = np.array(pil_image)

        # Convert RGB to BGR
        img = open_cv_image[:, :, ::-1].copy()

        input_points = get_rectangle(img)
        input_points = arrange_points(input_points)

        converted_points, height, width = set_four_points(input_points)
        matrix = cv2.getPerspectiveTransform(input_points, converted_points)

        img_output = cv2.warpPerspective(img, matrix, (width, height))

        color_converted = cv2.cvtColor(img_output, cv2.COLOR_BGR2RGB)

        pil_image = Image.fromarray(color_converted)

        pil_image.save(img_file)

        return pil_image


def union_boxes(boxes_xyxy_tensor):
    x_l_list, y_d_list, x_r_list, y_u_list = [], [], [], []
    for box in boxes_xyxy_tensor:
        x_l, y_d, x_r, y_u = box.detach().cpu().numpy()
        x_l_list.append(x_l)
        y_d_list.append(y_d)
        x_r_list.append(x_r)
        y_u_list.append(y_u)

    min_x_l, min_y_d = min(x_l_list), min(y_d_list)
    max_x_r, max_y_u = max(x_r_list), max(y_u_list)
    return [min_x_l, min_y_d, max_x_r, max_y_u]


def get_rectangle(img):
    img_shape = img.shape
    if len(img_shape) == 2:
        height, width = img_shape
    elif len(img_shape) == 3:
        height, width, channels = img_shape
    else:
        print(f'Unknown img shape: {img_shape}')

    left_upper = [0, 0]
    left_down = [0, height]
    right_down = [width, height]
    right_upper = [width, 0]
    return np.float32([left_upper, left_down, right_down, right_upper])


def arrange_points(points):
    # initialize a list of co-ordinates that will be ordered
    # first entry is top-left point, second entry is top-right
    # third entry is bottom-right, forth/last point is the bottom left point.
    rectangle = np.zeros((4, 2), dtype="float32")

    # bottom left point should be the smallest sum
    # the top-right point will have the largest sum of point.
    sum_points= points.sum(axis=1)
    rectangle[0] = points[np.argmin(sum_points)]
    rectangle[2] = points[np.argmax(sum_points)]


    #bottom right will have the smallest difference
    #top left will have the largest difference.
    diff_points = np.diff(points, axis=1)
    rectangle[1] = points[np.argmin(diff_points)]
    rectangle[3] = points[np.argmax(diff_points)]


    # return order of co-ordinates.
    return rectangle


def set_four_points(points):
    # obtain order of points and unpack.
    rectangle = arrange_points(points)
    (top_left,top_right,bottom_right,bottom_left) = rectangle
    # let's compute width of the rectangle.
    # using formular for distance between two points
    left_height = np.sqrt(((top_left[0]-bottom_left[0])**2) + ((top_left[1]-bottom_left[1])**2))
    right_height = np.sqrt(((top_right[0]-bottom_right[0])**2) + ((top_right[1]-bottom_right[1])**2))
    top_width = np.sqrt(((top_right[0]-top_left[0])**2) + ((top_right[1]-top_left[1])**2))
    bottom_width = np.sqrt(((bottom_right[0]-bottom_left[0])**2) + ((bottom_right[1]-bottom_left[1])**2))

    maxheight = max(int(left_height), int(right_height))
    maxwidth  = max(int(top_width), int(bottom_width))

    destination = np.array([
        [0,0],
        [maxwidth -1,0],
        [maxwidth -1, maxheight-1],
        [0, maxheight - 1]], dtype = "float32")
    return destination, maxheight, maxwidth


