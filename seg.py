import os
import pdal
import time
from segment_lidar import samlidar
import pandas as pd
import numpy as np

#Select model for segmenting

model = samlidar.SamLidar(ckpt_path="seg-lidar/sam_vit_h_4b8939.pth", algorithm='segment-geospatial')

# get pointcloud seed name
fileDir = input("Select the name of the folder in [./data/cloud] you would like to run: ")
prefix = "./data/class/"

#Throw an error if the requested data does not exist
if not os.path.exists(f"data/cloud/{fileDir}/"):
    raise FileNotFoundError("No data found for given seed")


#initialize directory locations
inDir = f"./data/cloud/{fileDir}/"
outRaster = f"{prefix}{fileDir}/rasters/"
outLabels = f"{prefix}{fileDir}/labels/"
outSeg = f"{prefix}{fileDir}/"
scattering = f"{prefix}{fileDir}/scattering/"
planarity = f"{prefix}{fileDir}/planarity/"
meanRet = f"{prefix}{fileDir}/meanRet/"
linearity = f"{prefix}{fileDir}/linearity/"
arrows = f"{prefix}{fileDir}/arrows/"
allData = f"{prefix}{fileDir}/all-info/"

#Create directories if they do not exist
li = ["./data", "./data/cloud", "./data/class", prefix+fileDir, outRaster, outLabels, outSeg, scattering, planarity, meanRet, linearity, arrows, allData]
for i in li:
    if not os.path.exists(i):
            os.mkdir(i)

#Initialize filters
hag = [pdal.Filter.hag_nn()]


for tile in os.listdir(inDir):
    #discards .DS_store in search
    if (not tile.startswith('.') and os.path.isfile(inDir + tile)):

        #obtain name for file
        file = tile.split('.')[0]

        start=time.time()
        print(inDir+tile)

        #read file for points
        pdal_points = model.read(inDir+tile)
        
        #performs segmentation on file
        pdal_points, noise = model.noiseFilter(pdal_points)
        cloud, non_ground, ground, pdal_points= model.smrf(pdal_points)


        #Add HAG
        pdal_points = model.applyFilters(pdal_points = pdal_points, filters = hag)

        #segment file
        labels, *_ = model.segment(points=cloud, image_path=outRaster+file+"-raster.tif", labels_path=outLabels+file+"-labeled.tif")
        points_grouped = model.grouping(pdal_points, labels, ground, non_ground, noise)
        pdal_points, bad_pts = model.featureFilter(points_grouped)

        #classify if there exist points
        if pdal_points is not None:
            classified_points = model.classify(pdal_points, bad_pts, file, fileDir)
            model.write_pdal(points=classified_points, segment_ids=labels, non_ground=non_ground, ground=ground, save_path=outSeg+file+"-segmented.copc.laz")

        end = time.time()
        print(f'Segment-lidar completed in {end - start:.2f} seconds.\n')
