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


parser = argparse.ArgumentParser(description='PyTorch Pré_Treined fasterrcnn')
#parser.add_argument("--dataset_path", default="", type=str, required=True,
 #                   help="informação.")
#parser.add_argument('--batch_size',
 #                   default=4, type=int, help='batch_size')
#parser.add_argument('--epoch', default=100,
 #                   type=int, help='you need in the epoch')
#parser.add_argument('--num-workers', type=int,
 #                   default=4, help='number of workers')
#parser.add_argument("--input_size", default=600,
 #                   type=int, help="input size img.")


args = parser.parse_args()

path_model = "/content/placas/15012024custon_fasterrcnn_mobilenet_v3_large_fpn.pth"
video_path = '/content/tokio_01_g.mp4'
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
    print(predictions)
 
 

if __name__ == '__main__':
    main()
