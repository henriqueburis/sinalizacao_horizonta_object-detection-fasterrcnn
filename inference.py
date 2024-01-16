#Buris L.H
import os
import torch
import torchvision
from torchvision import datasets, models

from torchvision.transforms import functional as F

import cv2

import argparse
import sys

# remove arnings (optional)
import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict, deque
import datetime
import time
from tqdm import tqdm # progress bar

from google.colab.patches import cv2_imshow

labels_ = ['frente', 'frente', 'frente_esqueda', 'frete_esqueda', 'frente',]

output_video = cv2.VideoWriter('/content/output.mp4', cv2.VideoWriter_fourcc(*'XVID'), 30, (1280, 720))  # 'output.mp4' é o nome do arquivo de saída MP4

path_model = "/content/placas/15012024custon_fasterrcnn_mobilenet_v3_large_fpn.pth"
video_path = '/content/video_teste.mp4'
n_classes = 5

def main():

  model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
  in_features = model.roi_heads.box_predictor.cls_score.in_features # we need to change the head
  model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, n_classes)
  model.load_state_dict(torch.load(path_model))
  model.eval()

  # Abra o vídeo
  cap = cv2.VideoCapture(video_path)

  #for frame in video:
  # Loop pelos quadros do vídeo
  while True:
    # Leia o próximo quadro
    ret, frame = cap.read()
    if not ret:
        break

    image_copy = frame.copy()

    #print(image_copy)

    # Converta o quadro em um tensor
    frame_tensor = F.to_tensor(frame).unsqueeze(0)

    # Faça uma detecção no quadro
    with torch.no_grad():
      predictions = model(frame_tensor.to('cpu'))
    #print(predictions)

    for score, bbox, labels in zip(predictions[0]['scores'], predictions[0]['boxes'], predictions[0]['labels']):
      if(score >= 0.8):
        x, y, w, h = bbox
        x, y, w, h = int(x), int(y), int(w), int(h)

        cv2.putText(image_copy, str(labels_[labels.item()]), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.rectangle(image_copy, (x, y), (w,h), (0, 0, 255), 2)


    output_video.write(image_copy)

  output_video.release()


if __name__ == '__main__':
    main()
