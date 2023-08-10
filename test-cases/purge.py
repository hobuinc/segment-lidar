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
             pdal.Filter.range(limits = "Classification[2:2]") 
    
    total_points = total_pipe.execute()
    ground_points = ground_pipe.execute()

    return total_points, ground_points

if __name__ == '__main__':
    #get directory for files from user
    location = input("Choose a folder to purge files from: ")
    location = "./data/cloud/" + location + "/"

    #raise error if location doesnt exist
    if not os.path.exists(location):
        raise FileNotFoundError('The folder you want to purge does not exist')

    for file_name in os.listdir(location):
        if not file_name.startswith('.'):
            print("Checking:", file_name)
            try:
                total, ground = num_points(location, file_name)
            except FileNotFoundError:
                print("File was not found")
        else:
            continue
        if (total - ground < 20000):
            os.remove(location + file_name)
    print('Finished purge')