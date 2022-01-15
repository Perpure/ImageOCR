from ts.torch_handler.base_handler import BaseHandler
import torch
import io
import cv2
import numpy as np
import math
from rec_postprocess import CTCLabelDecode


class RecHandler(BaseHandler):
    
    def __init__(self):
        self.rec_image_shape = (3, 32, 320)

        # TODO: Move to config
        cyr_dict = 'cyrillic_dict.txt'
        cyr_dict_to_rus = 'cyrillic_dict_to_rus.txt'
        lm_model = '/absolute/path/to/soc_arpa.binary'
        self.decode = CTCLabelDecode(cyr_dict, cyr_dict_to_rus, lm_model)
        super().__init__()

    def preprocess(self, data):
        """
        Preprocess function to convert the request input to a tensor(Torchserve supported format).
        The user needs to override to customize the pre-processing
        Args :
            data (list): List of the data from the request input.
        Returns:
            tensor: Returns the tensor data of the input
        """
        file = data[0]['file']
        inp = np.asarray(bytearray(file), dtype=np.uint8)
        img = cv2.imdecode(inp, cv2.IMREAD_COLOR)
        img = self.resize_norm_img(img)
        img = np.expand_dims(img, 0).astype(np.float32)
        return torch.as_tensor(img, device=self.device)

    def postprocess(self, data):
        """
        The post process function makes use of the output from the inference and converts into a
        Torchserve supported response output.
        Args:
            data (Torch Tensor): The torch tensor received from the prediction output of the model.
        Returns:
            List: The post process function returns a list of the predicted output.
        """
        return [self.decode(data.cpu())]

    def resize_norm_img(self, img):
        imgC, imgH, imgW = self.rec_image_shape
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im
