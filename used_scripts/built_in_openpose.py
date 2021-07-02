import cv2
import pandas as pd
import time
import numpy as np
from random import randint
import argparse
import sys, time
from openpose import pyopenpose as op
import os
parser = argparse.ArgumentParser(description='Run keypoint detection')
parser.add_argument("--device", default="gpu", help="Device to inference on")
parser.add_argument("--directory", default="/acer/Kinect_Data/Task by Task for data analysing(balance study)", help="Input Directory with files")
args = parser.parse_args()

output_directory= args.directory+"_openpose_csv_output"
#os.mkdir(output_directory)#create output directory

threshold = 0.2

key_points = {
    0:  "Nose", 1:  "Neck", 2:  "RShoulder", 3:  "RElbow", 4:  "RWrist", 5:  "LShoulder", 6:  "LElbow",
    7:  "LWrist", 8:  "MidHip", 9:  "RHip", 10: "RKnee", 11: "RAnkle", 12: "LHip", 13: "LKnee",
    14: "LAnkle", 15: "REye", 16: "LEye", 17: "REar", 18: "LEar", 19: "LBigToe", 20: "LSmallToe",
    21: "LHeel", 22: "RBigToe", 23: "RSmallToe", 24: "RHeel", 25: "Background"
}

#Body_25 keypoint pairs 
POSE_PAIRS = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],     #arm, shoulder line
            [1,8], [8,9], [9,10], [10,11], [8,12], [12,13], [13,14],  #2 leg
            [11,24], [11,22], [22,23], [14,21],[14,19],[19,20],    #2 foot  
            [1,0], [0,15], [15,17], [0,16], [16,18], #face
            [2,17], [5,18]
                ]



nPoints = 25
alpha = 0.3

directory = args.directory
out_dir=os.fsencode(output_directory)
##for all video files in the directory
for file in os.listdir(directory):
    filename = os.fsdecode(file) #get the filename
    output_csv_name=filename+"_csv_openpose_file.csv" #create an output csv file 
    outputPath=  output_directory +"/"+output_csv_name
    outputPath="%s" % outputPath

    if not filename.endswith(".mp4"):
        print(filename+"not a video")
    elif os.path.isfile(outputPath):
         print(output_csv_name+" already exists")
    else: #if it is a video file

        #start over with empty dataframe
        body_25_columns=[ "Nose","Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow",
                                                    "LWrist", "MidHip",  "RHip",  "RKnee", "RAnkle", "LHip",  "LKnee",
                                                    "LAnkle", "REye", "LEye", "REar", "LEar",  "LBigToe",  "LSmallToe",
                                                    "LHeel", "RBigToe", "RSmallToe",  "RHeel"]

        data = pd.DataFrame(columns=body_25_columns)
        video_path=video_path=directory+"/"+filename
        video_path="%s" % video_path
        print(video_path)
        cap = cv2.VideoCapture(video_path)
        ret, img = cap.read()
        if ret == False:
            print('Video File Read Error')
            sys.exit(0)
        frameHeight, frameWidth, c = img.shape

        frame = 0
        inHeight = 368
        t_elapsed = 0.0

        params = dict()
        params["display"] = "0"     #speed up the processing time
        params["render_pose"] = "0"

        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()


        while cap.isOpened():
            f_st = time.time()
            ret, img = cap.read()
            if ret == False:
                break
            frame += 1
            datum = op.Datum()
            datum.cvInputData = img
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))
            try:
                human_count = len(datum.poseKeypoints)
                print("Humans: "+ str(len(datum.poseKeypoints)))
                if human_count==1:
                    #print ("Body keypoints: \n" + str(datum.poseKeypoints[0]))
                    df2 = pd.DataFrame([datum.poseKeypoints[0].tolist() ], columns=body_25_columns)
                    data= data.append(df2,ignore_index=True)
                    #print(str(data))
            except TypeError:
                print("No Humans")
            except:
                print("Something else went wrong")
            f_elapsed = time.time() - f_st
            t_elapsed += f_elapsed
            print('Frame[%d] processed time[%4.2f]'%(frame, f_elapsed))
        data.to_csv(outputPath, index = False)

        print('Total processed time[%4.2f]'%(t_elapsed))
        print('avg frame processing rate :%4.2f'%(t_elapsed / frame))
        cap.release()
