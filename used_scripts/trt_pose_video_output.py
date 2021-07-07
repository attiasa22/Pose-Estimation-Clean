#!/usr/bin/python 
import json
import os
import argparse
import pandas as pd
import numpy as np
import trt_pose.coco
import trt_pose.models
import torch
import torch2trt
from torch2trt import TRTModule
import time
import cv2
import torchvision.transforms as transforms
import PIL.Image, PIL.ImageDraw
from trt_pose.parse_objects import ParseObjects
from trt_pose.draw_objects import DrawObjects


parser = argparse.ArgumentParser(description='Run keypoint detection')
parser.add_argument("--device", default="gpu", help="Device to inference on")
parser.add_argument("--directory", default="/acer/Kinect_Data/Task by Task for data analysing(balance study)", help="Input Directory with files")
args = parser.parse_args()

directory = args.directory
#Create the output directory name
output_directory= args.directory+"_trt_pose_csv_output"

#Load the points needed from the json file
with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

#Load the coco model
topology = trt_pose.coco.coco_category_to_topology(human_pose)

num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])

model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
MODEL_WEIGHTS = 'resnet18_baseline_att_224x224_A_epoch_249.pth'
model.load_state_dict(torch.load(MODEL_WEIGHTS))
#Width and height of images model was trained on
WIDTH = 224
HEIGHT = 224

data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()

#load torch model, takes some time to finish
print("Beginning to load model")
model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)
OPTIMIZED_MODEL='resnet18_baseline_att_224x224_A_epoch_249_trt.pth'

torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)

model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))
print("Finished")


#Lines 59-65 find the theoretical FPS your machine can run the model at. XAVIER runs at around 250 FPS
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

def draw_keypoints(img, key):
    thickness = 5
    w, h = img.size
    draw = PIL.ImageDraw.Draw(img)
     #draw Rankle -> RKnee (16-> 14)
    if all(key[16]) and all(key[14]):
        draw.line([ round(key[16][2] * w), round(key[16][1] * h), round(key[14][2] * w), round(key[14][1] * h)],width = thickness, fill=(51,51,204))
    #draw RKnee -> Rhip (14-> 12)
    if all(key[14]) and all(key[12]):
        draw.line([ round(key[14][2] * w), round(key[14][1] * h), round(key[12][2] * w), round(key[12][1] * h)],width = thickness, fill=(51,51,204))
    #draw Rhip -> Lhip (12-> 11)
    if all(key[12]) and all(key[11]):
        draw.line([ round(key[12][2] * w), round(key[12][1] * h), round(key[11][2] * w), round(key[11][1] * h)],width = thickness, fill=(51,51,204))
    #draw Lhip -> Lknee (11-> 13)
    if all(key[11]) and all(key[13]):
        draw.line([ round(key[11][2] * w), round(key[11][1] * h), round(key[13][2] * w), round(key[13][1] * h)],width = thickness, fill=(51,51,204))
    #draw Lknee -> Lankle (13-> 15)
    if all(key[13]) and all(key[15]):
        draw.line([ round(key[13][2] * w), round(key[13][1] * h), round(key[15][2] * w), round(key[15][1] * h)],width = thickness, fill=(51,51,204))

    #draw Rwrist -> Relbow (10-> 8)
    if all(key[10]) and all(key[8]):
        draw.line([ round(key[10][2] * w), round(key[10][1] * h), round(key[8][2] * w), round(key[8][1] * h)],width = thickness, fill=(255,255,51))
    #draw Relbow -> Rshoulder (8-> 6)
    if all(key[8]) and all(key[6]):
        draw.line([ round(key[8][2] * w), round(key[8][1] * h), round(key[6][2] * w), round(key[6][1] * h)],width = thickness, fill=(255,255,51))
    #draw Rshoulder -> Lshoulder (6-> 5)
    if all(key[6]) and all(key[5]):
        draw.line([ round(key[6][2] * w), round(key[6][1] * h), round(key[5][2] * w), round(key[5][1] * h)],width = thickness, fill=(255,255,0))
    #draw Lshoulder -> Lelbow (5-> 7)
    if all(key[5]) and all(key[7]):
        draw.line([ round(key[5][2] * w), round(key[5][1] * h), round(key[7][2] * w), round(key[7][1] * h)],width = thickness, fill=(51,255,51))
    #draw Lelbow -> Lwrist (7-> 9)
    if all(key[7]) and all(key[9]):
        draw.line([ round(key[7][2] * w), round(key[7][1] * h), round(key[9][2] * w), round(key[9][1] * h)],width = thickness, fill=(51,255,51))

    #draw Rshoulder -> RHip (6-> 12)
    if all(key[6]) and all(key[12]):
        draw.line([ round(key[6][2] * w), round(key[6][1] * h), round(key[12][2] * w), round(key[12][1] * h)],width = thickness, fill=(153,0,51))
    #draw Lshoulder -> LHip (5-> 11)
    if all(key[5]) and all(key[11]):
        draw.line([ round(key[5][2] * w), round(key[5][1] * h), round(key[11][2] * w), round(key[11][1] * h)],width = thickness, fill=(153,0,51))


    #draw nose -> Reye (0-> 2)
    if all(key[0][1:]) and all(key[2]):
        draw.line([ round(key[0][2] * w), round(key[0][1] * h), round(key[2][2] * w), round(key[2][1] * h)],width = thickness, fill=(219,0,219))

    #draw Reye -> Rear (2-> 4)
    if all(key[2]) and all(key[4]):
        draw.line([ round(key[2][2] * w), round(key[2][1] * h), round(key[4][2] * w), round(key[4][1] * h)],width = thickness, fill=(219,0,219))

    #draw nose -> Leye (0-> 1)
    if all(key[0][1:]) and all(key[1]):
        draw.line([ round(key[0][2] * w), round(key[0][1] * h), round(key[1][2] * w), round(key[1][1] * h)],width = thickness, fill=(219,0,219))

    #draw Leye -> Lear (1-> 3)
    if all(key[1]) and all(key[3]):
        draw.line([ round(key[1][2] * w), round(key[1][1] * h), round(key[3][2] * w), round(key[3][1] * h)],width = thickness, fill=(219,0,219))

    #draw nose -> neck (0-> 17)
    if all(key[0][1:]) and all(key[17]):
        draw.line([ round(key[0][2] * w), round(key[0][1] * h), round(key[17][2] * w), round(key[17][1] * h)],width = thickness, fill=(255,255,0))
    return img

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

# UNUSED
#def execute(resizedImage,img):
#    frameData=[]
#    X_compress = 640.0 / WIDTH * 1.0
#    Y_compress = 480.0 / HEIGHT * 1.0
#    image = img
#    parse_objects = ParseObjects(topology)
#    data = preprocess(image)
#    cmap, paf = model_trt(data)
#    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
#    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
#    for i in range(counts[0]):
#        print(i+" humans")
#        keypoints = get_keypoint(object, i, peaks)
#        for j in range(len(keypoints)):
#            if keypoints[j][1]:
#                x = round(keypoints[j][2] * WIDTH * X_compress)
#                y = round(keypoints[j][1] * HEIGHT * Y_compress)
#                frameData.append([x,y])
#        return frameData
parse_objects = ParseObjects(topology)
def execute_2(img, org, count):
    start = time.time()
    data = preprocess(img)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    end = time.time()
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
    for i in range(counts[0]):
        #print("Human index:%d "%( i ))
        kpoint = get_keypoint(objects, i, peaks)
        #print(kpoint)
        org = draw_keypoints(org, kpoint)
    netfps = 1 / (end - start)
    draw = PIL.ImageDraw.Draw(org)
   # draw.text((30, 30), "NET FPS:%4.1f"%netfps, font=fnt, fill=(0,255,0))    
    print("Human count:%d len:%d "%(counts[0], len(counts)))
    print('===== Frmae[%d] Net FPS :%f ====='%(count, netfps))
    return kpoint, org

for file in os.listdir(directory):
    filename = os.fsdecode(file) #get the filename
    output_csv_name=filename+"_csv_trt_pose_file.csv" #create an output csv file 
    outputPath=  output_directory +"/"+output_csv_name
    outputPath="%s" % outputPath
    out_video_filename= 'TRT_%s.mp4'%filename
    if not filename.endswith(".mp4"):
        print(filename+" not a video")
    elif os.path.isfile("trt_videos/%s"%out_video_filename):
        print(output_csv_name+" already exists")
    else: #if it is a video file


      #  data = pd.DataFrame(columns=human_pose['keypoints'])
        video_path=video_path=directory+"/"+filename
        video_path="%s" % video_path

        print(video_path)
        cap = cv2.VideoCapture(video_path)
        ret, img = cap.read()
        if ret == False:
            print('Video File Read Error')

        H, W, __ = img.shape
        frame=0
        t_elapsed = 0.0

        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        out_video = cv2.VideoWriter(out_video_filename, fourcc, cap.get(cv2.CAP_PROP_FPS), (W, H))
        os.rename(out_video_filename,"/acer/trt_videos/%s"%out_video_filename)
        print('/acer/trt_videos/TRT_%s'%(filename))
        draw_objects = DrawObjects(topology)
        while cap.isOpened():
            f_st = time.time()
            frame+=1
            ret, img = cap.read()
            if ret == False:
                break

            resizedImage=cv2.resize(img, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
            pilimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pilimg = PIL.Image.fromarray(pilimg)
            frameData, drawnFrame=execute_2(resizedImage,pilimg,frame)
            print("Clean DATA")
                #print(frameData) 
                #print("END Clean DATA")
            f_elapsed = time.time() - f_st
            t_elapsed += f_elapsed
            print('Frame[%d] processed time[%4.2f]'%(frame, f_elapsed))
     #       df2 = pd.DataFrame([frameData], columns=human_pose['keypoints'])
     #       data= data.append(df2,ignore_index=True)
          #  print(data)
            array = np.asarray(pilimg, dtype="uint8")
            out_video.write(array)

    #     data.to_csv(outputPath, index = False)
        cap.release()
        out_video.release()

