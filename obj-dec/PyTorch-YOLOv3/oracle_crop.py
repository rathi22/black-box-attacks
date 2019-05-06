from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

def oracle(dataloader):
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    # parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    # parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    # parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    # parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    # parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    # parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    # parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    # parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    # parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    # parser.add_argument("--yolo",  help="path to checkpoint model")
    # opts = parser.parse_args()
    # print(opt)
    base_path = "./obj-dec/PyTorch-YOLOv3/"
    opt = {}
    opt["model_def"] = base_path+"config/yolov3.cfg"
    opt["img_size"] = 416
    opt["weights_path"] = base_path+"weights/yolov3.weights"
    opt["class_path"] = base_path+"data/coco.names"
    opt["conf_thres"] = 0.8 
    opt["nms_thres"] = 0.4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt["model_def"], img_size=opt["img_size"]).to(device)

    if opt["weights_path"].endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt["weights_path"])
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt["weights_path"]))

    model.eval()  # Set in evaluation mode

    # dataloader = DataLoader(
    #     ImageFolder(opt[image_folder, img_size=opt[img_size),
    #     batch_size=opt[batch_size,
    #     shuffle=False,
    #     num_workers=opt[n_cpu,
    # )

    classes = load_classes(opt["class_path"])  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    answer = []

    for batch_i, (input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = input_imgs[0]
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, opt["conf_thres"], opt["nms_thres"])
            print(detections)
            if detections[0] is None:
                answer.append([80]) # as of now
            else:
                max_size = 0
                max_class = -1
                for tup in detections[0]:
                    sz = (tup[2]-tup[0])*(tup[3]-tup[1])
                    if sz > max_size:
                        max_size = sz
                        max_class = int(tup[-1])
                answer.append([max_class])
    
    return torch.tensor(answer)