import requests
import json
from PIL import Image
from draw_ocr import draw_ocr

inference_url = 'http://localhost:8786'
img_file = 'img/0.jpg'
file = {'file': ('image.jpg', open(img_file, 'rb'), 'image/jpeg')}
res = requests.post(url=inference_url + '/predictions/TextDetection', files=file)
out = json.loads(res.text)

boxes = list(map(lambda x: x[0], out))
texts = list(map(lambda x: x[1], out))

image = Image.open(img_file).convert('RGB')
im_show = draw_ocr(image, boxes, txts=texts, font_path='ocr_metrics/data/fonts/Verdana.ttf')
im_show = Image.fromarray(im_show)
im_show.save('img/result.jpg')
