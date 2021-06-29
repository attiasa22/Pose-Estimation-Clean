# Imprt Libraries
from scipy.spatial import distance as dist
import numpy as np
import pandas as pd
import cv2
import os
import sys

protoFile = "~/Desktop/openpose/models/pose/body_25/pose_deploy.prototxt"
weightsFile = "~/Desktop/openpose/models/pose/body_25/pose_iter_584000.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
def setup_video(video_path):
    # Store the input video specifics
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    ok, frame = cap.read()
    (frameHeight, frameWidth) = frame.shape[:2]
    h = 500
    w = int((h/frameHeight) * frameWidth)

    # Dimensions for inputing into the model
    inHeight = 368
    inWidth = 368
    return cap, n_frames, fps, ok, frame, (frameHeight, frameWidth), h,w, inHeight,inWidth


def extractData(directory_in_str,output_directory):
    directory = os.fsencode(directory_in_str)
    out_dir=os.fsencode(output_directory)
    ##for all video files in the directory
    for file in os.listdir(directory):
        filename = os.fsdecode(file) #get the filename
        if filename.endswith(".mp4"): #if it is a video file
            output_csv_name=filename+"_csv_openpose_file.csv" #create an output csv file as well as a dataframe

            data = pd.DataFrame(data=None,columns=[ "Nose","Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow",
                                                    "LWrist", "MidHip",  "RHip",  "RKnee", "RAnkle", "LHip",  "LKnee",
                                                    "LAnkle", "REye", "LEye", "REar", "LEar",  "LBigToe",  "LSmallToe",
                                                    "LHeel", "RBigToe", "RSmallToe",  "RHeel","Background"])
            
            file_path=os.path.join(directory, filename)
            print(os.path.join(directory, filename))

            cap, n_frames, fps, ok, frame, (frameHeight, frameWidth), h,w, inHeight,inWidth=setup_video(file_path)
            #body_25 model has 25 (x,y) points
            previous_points = [(0,0),(0,0),(0,0),(0,0),(0,0),
                                      (0,0),(0,0),(0,0),(0,0),(0,0),
                                      (0,0),(0,0),(0,0),(0,0),(0,0),
                                      (0,0),(0,0),(0,0),(0,0),(0,0),
                                      (0,0),(0,0),(0,0),(0,0),(0,0)]

            thresh = 0.1 
            while True:
                ok, frame = cap.read()

                if ok != True:
                    break
                
                frame = cv2.resize(frame, (w, h), cv2.INTER_AREA)    
                frame_copy = np.copy(frame)
                
                # Input the frame into the model
                inpBlob = cv2.dnn.blobFromImage(frame_copy, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
                net.setInput(inpBlob)
                output = net.forward()
                
                H = output.shape[2]
                W = output.shape[3]
                
                curr_data = []
                
                # Iterate through the returned output and store the data
                # Iterate through the returned output and store the data
                for i in range(25):
                    probMap = output[0, i, :, :]
                    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
                    x = (w * point[0]) / W
                    y = (h * point[1]) / H
                    
                    if prob > thresh:
                        curr_data.append((int(x), int(y)))
                    else :
                        curr_data.append(previous_points[i])
                
                data.append(curr_data)
                previous_points = curr_data
                
                key = cv2.waitKey(1) & 0xFF
            
                if key == ord("q"):
                    break
            # Save the output data from the video in CSV format
            df = pd.DataFrame(data)
            df.columns = ['a', 'b']
            df.to_csv(output_csv_name, index = False)
            print('save complete')

            



    #create output csv
    #Each frame should be a row
    #Columns: each points x,y, confidence value

    #FIND THE POINTS for each fram and input
#output csv file with correct name 


if __name__ == "__main__":
    directory_in_str= str(sys.argv[1]) #pass directory as argument
    output_directory= directory_in_str+"_openpose_csv_output"
    os.mkdir(output_directory)#create output directory
    extractData(directory_in_str,output_directory)