#!/usr/bin/python 
import json
import os
import argparse
import numpy as np
import pandas as pd
import sys
#import trt_pose
import trt_pose.coco
import trt_pose.models
import torch
import torch2trt
from torch2trt import TRTModule
import time
import cv2
import torchvision.transforms as transforms
import PIL.Image
from trt_pose.parse_objects import ParseObjects

parser = argparse.ArgumentParser(description='Run keypoint detection')
parser.add_argument("--device", default="gpu", help="Device to inference on")
parser.add_argument("--directory", default="/acer/Kinect_Data/Task by Task for data analysing(balance study)", help="Input Directory with files")
args = parser.parse_args()

output_directory= args.directory+"_trt_pose_csv_output"
directory = args.directory
out_dir=os.fsencode(output_directory)




with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)
print("1")
num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])

model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
print("2")
MODEL_WEIGHTS = 'resnet18_baseline_att_224x224_A_epoch_249.pth'
model.load_state_dict(torch.load(MODEL_WEIGHTS))

WIDTH = 224
HEIGHT = 224
print("3")
data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()

model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)

OPTIMIZED_MODEL='resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
print("4")
torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)

model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))
print("5")



t0 = time.time()
torch.cuda.current_stream().synchronize()
for i in range(50):
    y = model_trt(data)
torch.cuda.current_stream().synchronize()
t1 = time.time()
print(50.0 / (t1 - t0))


mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

def get_keypoint(humans, hnum, peaks):
    #check invalid human index
    kpoint = []
    human = humans[0][hnum]
    C = human.shape[0]
    for j in range(C):
        k = int(human[j])
        if k >= 0:
            peak = peaks[0][j][k]   # peak[1]:width, peak[0]:height
            peak = (j, float(peak[0]), float(peak[1]))
            kpoint.append(peak)
            #print('index:%d : success [%5.3f, %5.3f]'%(j, peak[1], peak[2]) )
        else:
            peak = (j, None, None)
            kpoint.append(peak)
            #print('index:%d : None %d'%(j, k) )
    return kpoint

def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

parse_objects = ParseObjects(topology)
def execute(resizedImage,img):
    frameData=[]
    X_compress = 640.0 / WIDTH * 1.0
    Y_compress = 480.0 / HEIGHT * 1.0
    image = img
    data = preprocess(image)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
    for i in range(counts[0]):
        print(i+" humans")
        keypoints = get_keypoint(object, i, peaks)
        for j in range(len(keypoints)):
            if keypoints[j][1]:
                x = round(keypoints[j][2] * WIDTH * X_compress)
                y = round(keypoints[j][1] * HEIGHT * Y_compress)
                frameData.append([x,y])
        return frameData

for file in os.listdir(directory):
    filename = os.fsdecode(file) #get the filename
    output_csv_name=filename+"_csv_trt_pose_file.csv" #create an output csv file 
    outputPath=  output_directory +"/"+output_csv_name
    outputPath="%s" % outputPath

    if not filename.endswith(".mp4"):
        print(filename+"not a video")
    elif os.path.isfile(outputPath):
         print(output_csv_name+" already exists")
    else: #if it is a video file

        data = pd.DataFrame(columns=human_pose['keypoints'])
        video_path=video_path=directory+"/"+filename
        video_path="%s" % video_path
        print(video_path)
        cap = cv2.VideoCapture(video_path)
        ret, img = cap.read()
        if ret == False:
            print('Video File Read Error')
            sys.exit(0)
        frame=0
        t_elapsed = 0.0
        while cap.isOpened():
            f_st = time.time()
            frame+=1
            ret, img = cap.read()
            if ret == False:
                break

            resizedImage=cv2.resize(img, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)

            frameData=execute(resizedImage,img)
            print(frameData)
            f_elapsed = time.time() - f_st
            t_elapsed += f_elapsed
            print('Frame[%d] processed time[%4.2f]'%(frame, f_elapsed))
            df2 = pd.DataFrame([frameData[0]], columns=human_pose['keypoints'])
            data= data.append(df2,ignore_index=True)

        data.to_csv(outputPath, index = False)
        cap.release()
