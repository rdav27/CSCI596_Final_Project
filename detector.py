from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
import argparse
import os
import os.path as osp
from darknet import DarkNet
import pickle as pkl
import pandas as pd
import random


# Parse arguments to the detect module
def arg_parse():
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

    parser.add_argument("--images", dest='images', help="Image / Directory containing images to perform detection upon",
                        default="imgs", type=str)
    parser.add_argument("--det", dest='det', help="Image / Directory to store detections to", default="det", type=str)
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help="Config file", default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help="weightsfile", default="yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso',
                        help="Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)

    return parser.parse_args()


args = arg_parse()
images = args.images
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()

# Loading the network
num_classes = 80  # For COCO
classes = load_classes("data/coco.txt")

# Set up the neural network
print("Loading network.....")
model = DarkNet(args.cfgfile)
model.load_weights(args.weightsfile)
print("Network successfully loaded")

model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0
assert inp_dim > 32

# If there's a GPU available, put the model on GPU
if CUDA:
    model.cuda()

model.eval()

read_dir = time.time()  # Checkpoint used to measure time

# Read the input images
try:
    imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
except NotADirectoryError:
    imlist = []
    imlist.append(osp.join(osp.realpath('.'), images))
except FileNotFoundError:
    print("No file or directory with the name {}".format(images))
    exit()

if not os.path.exists(args.det):
    os.makedirs(args.det)

load_batch = time.time()  # Checkpoint used to measure time
loaded_ims = [cv2.imread(x) for x in imlist]  # Use OpenCV to load the images

# OpenCV loads an image as a numpy array, with BGR as the order of the color channels.
# PyTorch's image input format is (Batches x Channels x Height x Width), with the channel order being RGB.
# Therefore, we need to transform the numpy array into PyTorch's input format.

# PyTorch Variables for images
im_batches = list(map(prep_image, loaded_ims, [inp_dim for x in range(len(imlist))]))

# List containing dimensions of original images
im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

if CUDA:
    im_dim_list = im_dim_list.cuda()

# Create the batches
leftover = 0
if len(im_dim_list) % batch_size:
    leftover = 1

if batch_size != 1:
    num_batches = len(imlist) // batch_size + leftover
    im_batches = [torch.cat((im_batches[i * batch_size: min((i + 1) * batch_size, len(im_batches))])) for i in
                  range(num_batches)]

# The detection loop
write = 0
start_det_loop = time.time()
for i, batch in enumerate(im_batches):
    # load the image
    start = time.time()
    if CUDA:
        batch = batch.cuda()

    prediction = model(Variable(batch, volatile=True), CUDA)

    prediction = write_results(prediction, confidence, num_classes, nms_conf=nms_thesh)

    end = time.time()

    if type(prediction) == int:
        for im_num, image in enumerate(imlist[i * batch_size: min((i + 1) * batch_size, len(imlist))]):
            im_id = i * batch_size + im_num
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start) / batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", ""))
            print("----------------------------------------------------------")
        continue

    prediction[:, 0] += i * batch_size  # transform the attribute from index in batch to index in imlist

    if not write:  # If we haven't initialised output
        output = prediction
        write = 1
    else:
        output = torch.cat((output, prediction))

    for im_num, image in enumerate(imlist[i * batch_size: min((i + 1) * batch_size, len(imlist))]):
        im_id = i * batch_size + im_num
        objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
        print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start) / batch_size))
        print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
        print("----------------------------------------------------------")

    if CUDA:
        # Makes sure that CUDA kernel is synchronized with the CPU.
        # Otherwise, CUDA kernel returns the control to CPU as soon as the GPU job is queued and well before the GPU job is completed (Asynchronous calling)
        torch.cuda.synchronize()

try:
    output
except NameError:
    print("No detections were made")
    exit()

im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())

scaling_factor = torch.min(inp_dim / im_dim_list, 1)[0].view(-1, 1)

output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2

output[:, 1:5] /= scaling_factor

for i in range(output.shape[0]):
    output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim_list[i, 0])
    output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim_list[i, 1])

output_recast = time.time()

class_load = time.time()
colors = pkl.load(open("pallete", "rb"))

draw = time.time()


# Draws a rectangle with a color of a random choice from colors, and writes the class of the object detected
def write(x, results, r_colors):
    c1 = tuple(x[1:3].cpu().int().numpy())
    c2 = tuple(x[3:5].cpu().int().numpy())
    img = results[int(x[0])]
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    color = random.choice(r_colors)
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
    return img


list(map(lambda x: write(x, loaded_ims, colors), output))
det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(args.det, x.split("\\")[-1]))
list(map(cv2.imwrite, det_names, loaded_ims))
end = time.time()

print("SUMMARY")
print("----------------------------------------------------------")
print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
print()
print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) + " images)", output_recast - start_det_loop))
print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch) / len(imlist)))
print("----------------------------------------------------------")

torch.cuda.empty_cache()
