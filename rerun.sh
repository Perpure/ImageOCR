#!/usr/bin/env bash

torchserve --stop

torch-model-archiver --model-name TextDetection --serialized-file det.pt --handler det_handler.py \
--extra-files det_postprocess.py,det_sort_boxes.py,config.yaml -f -v 0.1

torch-model-archiver --model-name TextRecognition --serialized-file rec.pt --handler rec_handler.py \
--extra-files rec_postprocess.py,dicts/cyrillic_dict.txt,dicts/cyrillic_dict_to_rus.txt,config.yaml -f -v 0.1

torchserve --start --ncs --model-store ./ \
--models TextDetection=TextDetection.mar TextRecognition=TextRecognition.mar --ts-config ts_config