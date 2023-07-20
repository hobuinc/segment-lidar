import random
import sys
import os
import numpy as np
import pdal
import pyproj
import time

def num_points(location, file_name):
    """performs the smrf algorithm on each file"""
    file_location = location + file_name
    reader = pdal.Reader.copc(file_location) | \
             pdal.Filter.assign(value = "Classification = 0") | \
             pdal.Filter.assign(value = "ReturnNumber = 1 WHERE ReturnNumber < 1") | \
             pdal.Filter.assign(value = "NumberOfReturns = 1 WHERE NumberOfReturns < 1")
             
    
    total_pipe = reader | pdal.Filter.smrf()
    
    ground_pipe = reader | pdal.Filter.smrf() | \
             pdal.Filter.range(limits = "Classification[2:2]") | \
             pdal.Writer.copc(f"./data/cloud/ground/{file_name}-filter.copc.laz",
                              forward="all")
    
    total_points = total_pipe.execute()
    ground_points = ground_pipe.execute()
    #breakpoint() 

    return total_points, ground_points

if __name__ == '__main__':
    #get directory for files from user
    location = input("Choose a folder to purge files from: ")
    print(location)
    location = "./data/cloud/" + location + "/"

    #make directory for saving files
    if not os.path.exists("./data/cloud/ground/"):
        os.mkdir("./data/cloud/ground/")

    if not os.path.exists(location):
        os.mkdir(location)

    for file_name in os.listdir(location):
        if not file_name.startswith('.'):
            #breakpoint()
            print("Checking:", file_name)
            try:
                total, ground = num_points(location, file_name)
            except FileNotFoundError:
                print("File was not found")
        else:
            continue
        num_air = total - ground
        if (total - ground < 20000):
            os.remove(location + file_name)