import copy
import os.path

from PIL import Image
from ts.torch_handler.base_handler import BaseHandler
import torch
import json
import cv2
import numpy as np
from det_postprocess import DBPostProcess
import requests
from det_sort_boxes import sort_boxes

class DetHandler(BaseHandler):
    
    def __init__(self):
        self.get_boxes = DBPostProcess()
        self.rec_url = 'http://localhost:8786/predictions/TextRecognition'
        self.workdir = '/path/to/your/workdir' # used for debug
        super().__init__()

    def preprocess(self, data):
        """
        Preprocess function to convert the request input to a tensor(Torchserve supported format).
        Args :
            data (list): List of the data from the request input.
        Returns:
            tensor: Returns the tensor data of the input
        """
        file = data[0]['file']
        inp = np.asarray(bytearray(file), dtype=np.uint8)
        img = cv2.imdecode(inp, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.orig_image = img.copy()
        self.ori_h, self.ori_w, _ = img.shape
        img, [self.ratio_h, self.ratio_w] = self.resize_image(img)

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        scale = 1. / 255
        norm_img = (img * scale - mean) / std

        transpose_img = norm_img.transpose(2, 0, 1)
        transpose_img = np.expand_dims(transpose_img, 0).astype(np.float32)

        return torch.as_tensor(transpose_img, device=self.device)

    def postprocess(self, data):
        """
        The post process function makes use of the output from the inference and converts into a
        Torchserve supported response output.
        Args:
            data (Torch Tensor): The torch tensor received from the prediction output of the model.
        Returns:
            List: The post process function returns a list of the predicted output.
        """
        shape_list = [[self.ori_h, self.ori_w, self.ratio_h, self.ratio_w]]

        data = data.cpu().numpy()
        # self.draw_mask(data)

        boxes = self.get_boxes(data, shape_list)[0]['points'].astype(np.float32)
        boxes = self.filter_tag_det_res_only_clip(boxes, (self.ori_h, self.ori_w))

        boxes = self.pre_sort_boxes(boxes)
        boxes = sort_boxes(boxes)

        response = []
        for i in range(len(boxes)):
            tmp_box = copy.deepcopy(boxes[i])
            img = self.get_rotate_crop_image(self.orig_image, tmp_box)
            response.append((boxes[i].tolist(), self.ocr_img(img)))

        return [response]

    def draw_mask(self, data):
        mask_img = cv2.resize(self.orig_image, data.shape[:1:-1])
        for i in range(len(data[0][0])):
            for j in range(len(data[0][0][i])):
                if data[0][0][i][j] > 0.1:
                    mask_img[i][j][0] = 255
                    mask_img[i][j][1] //= 2
                    mask_img[i][j][2] //= 2
        mask_img = cv2.resize(mask_img, (self.ori_w, self.ori_h))
        Image.fromarray(mask_img).save(os.path.join(self.workdir, 'img/mask.jpg'))

    def ocr_img(self, img):
        imencoded = cv2.imencode(".jpg", img)[1]
        file = {'file': ('image.jpg', imencoded.tostring(), 'image/jpeg')}
        response = requests.post(self.rec_url, files=file)
        return json.loads(response.text)[0]

    def order_points_clockwise(self, pts):
        xSorted = pts[np.argsort(pts[:, 0]), :]

        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost

        rect = np.array([tl, tr, br, bl], dtype="float32")
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def resize_image(self, img):
        """
        resize image to a size multiple of 32 which is required by the network
        args:
            img(array): array with shape [h, w, c]
        return(tuple):
            img, (ratio_h, ratio_w)
        """

        # TODO: move to config
        limit_side_len = 1216
        limit_type = 'min'

        h, w, _ = img.shape

        # limit the max side
        if limit_type == 'max':
            if max(h, w) > limit_side_len:
                if h > w:
                    ratio = float(limit_side_len) / h
                else:
                    ratio = float(limit_side_len) / w
            else:
                ratio = 1.
        else:
            if min(h, w) < limit_side_len:
                if h < w:
                    ratio = float(limit_side_len) / h
                else:
                    ratio = float(limit_side_len) / w
            else:
                ratio = 1.
        resize_h = int(h * ratio)
        resize_w = int(w * ratio)

        resize_h = max(int(round(resize_h / 32) * 32), 32)
        resize_w = max(int(round(resize_w / 32) * 32), 32)

        if int(resize_w) <= 0 or int(resize_h) <= 0:
            return None, (None, None)

        img = cv2.resize(img, (int(resize_w), int(resize_h)))
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        return img, [ratio_h, ratio_w]
    
    def get_rotate_crop_image(self, img, points):
        '''
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        '''
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img


    def pre_sort_boxes(self, dt_boxes):
        return sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
