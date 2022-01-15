import os
import requests
import numpy as np
from asrtoolkit import cer, wer
import json
import time

data = 'data/eval_img/'
n_files = 1000

cers, wers = np.array([]), np.array([])
cers_all, wers_all = np.array([]), np.array([])
anss = []
n_missed = 0


def ocr(img):
    inference_url = 'http://127.0.0.1:8786'
    file = {'file': ('image.jpg', open(img, 'rb'), 'image/jpeg')}
    res = requests.post(url=inference_url + '/predictions/TextDetection', files=file)
    out = json.loads(res.text)
    return ' '.join(list(map(lambda x: x[1], out)))


with open('results.txt', 'w') as f:
    for i in range(n_files):
        print(i)
        print()
        with open(data + str(i) + '.txt', 'r') as source:
            target = source.read()
        if not ' ' in target:
            continue
        image = os.path.join(data, str(i) + '.jpg')

        pred = ocr(image)

        if pred == '':
            n_missed += 1

        pred = pred.lower()
        target = target.lower()


        werr = wer(pred, target)
        wers = np.append(wers, werr)
        cerr = cer(pred, target)
        cers = np.append(cers, cerr)
        f.write('{} \'{}\' \'{}\' {} {}\n'.format(i, pred, target, round(werr, 2), round(cerr, 2)))

    f.write('overall: cer={}, wer={}\n'.format(round(cers.mean(), 2), round(wers.mean(), 2)))
    f.write('n_misses={}\n'.format(n_missed))

