import os
import pdal
import time
from segment_lidar import samlidar
import pandas as pd
import numpy as np

#Select model for segmentings
model = samlidar.SamLidar(ckpt_path="seg-lidar/sam_vit_h_4b8939.pth")

inDir = "tiles/"
outRaster = "rasters/"
outLabels = "labels/"
outSeg = "segments/"

#Create directories if they do not exist
li = [inDir, outRaster, outLabels, outSeg]
for i in li:
    if not os.path.exists(i):
        os.mkdir(i)

for tile in os.listdir(inDir):
    #discards .DS_store
    if (not tile.startswith('.')):

        #obtain name for file
        file = os.path.splitext(tile)[0]
        start=time.time()
        print(inDir+tile)

        #read file for points
        pdal_points = model.read(inDir+tile)
        
        #performs segmentation on file
        pdal_points, noise = model.noiseFilter(pdal_points)
        cloud, non_ground, ground, pdal_points= model.smrf(pdal_points)
        labels, *_ = model.segment(points=cloud, image_path=outRaster+tile+"-raster.tif", labels_path=outLabels+tile+"-labeled.tif")
        pdal_points = model.applyFilters(pdal_points, [hag])
        points_grouped = model.grouping(pdal_points, labels, ground, non_ground)
        pdal_points, bad_pts = model.featureFilter(points_grouped)
        pdal_points = model.classify(pdal_points, bad_pts)
        model.write_pdal(points=pdal_points, segment_ids=labels, non_ground=non_ground, ground=ground, save_path=outSeg+file+"segmented.copc.laz")
        end = time.time()
        print(f'Segment-lidar completed in {end - start:.2f} seconds.\n')
