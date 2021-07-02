import os
import argparse
import pandas as pd
import re

parser = argparse.ArgumentParser(description='Combine CSV files')
parser.add_argument("--old_directory", default="/acer/Kinect_Data/Task by Task for data analysing(balance study))_trt_pose_csv_output", help="Input Directory with files")
parser.add_argument("--new_directory", default="/acer/Kinect_Data/Combined_CSVs", help="Output Directory with files organized by person's data")
args = parser.parse_args()

#Given old directory and new directory
#Go through each file in old directory
oldDirectory = args.old_directory
newDirectory = args.new_directory
for file in os.listdir(oldDirectory):
    filename = os.fsdecode(file) #get the filename
    #convert csv to dataframes
    taskData= pd.read_csv(filename)

    initials,taskNumber=findInitialsTask(filename)
    print(initials+", task: " + taskNumber)

    taskData.insert(0,column = "Task",value=taskNumber)

    # Combine all necessary csvs
    #is else statement necessary?
    if os.path.isfile(newDirectory+'/'+initials):
        taskData.to_csv('my_csv.csv', mode='a', header=False)
    else:
        taskData.to_csv(newDirectory+'/'+initials, index = False)

def findInitialsTask(filename):
    initials="0"
    task="-1"
    filenameValues= filename.split("_")
   
    for value in filenameValues:
        if len(value)==2:
            initials=value
        taskIndex= filename.lower().find("task")
        if taskIndex != -1:
            task= re.sub("_",'',value[taskIndex+4:taskIndex+6])
            
    return initials,task