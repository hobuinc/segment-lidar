import os
from posixpath import splitext
import pdal
import time
from segment_lidar import samlidar
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.feather as ft

#Select model for segmentings
model = samlidar.SamLidar(ckpt_path="seg-lidar/sam_vit_h_4b8939.pth")
#initialize directory location
inDir = "tiles/"
outRaster = "rasters/"
outLabels = "labels/"
outSeg = "segments/"
arrows = "arrows/"

#Create directories if they do not exist
li = [inDir, outRaster, outLabels, outSeg]
for i in li:
    try:
        os.path.exists(i)
    except:
        os.mkdir(i)

#Initialize filters
sort = pdal.Filter.sort(dimension="GPSTime")
outlier = pdal.Filter.outlier(method="radius", radius="1.0", min_k="4")
filters = [outlier, sort]

#obtains feathers containing pdal_points and classifies them
if 'points_filtered.feather' in os.listdir(arrows):
    feather = ft.read_table('arrows/points_filtered.feather')
    df1 = feather.to_pandas()
    s = df1.dtypes
    pdal_points = np.array([tuple(x) for x in df1.values], dtype=list(zip(s.index, s)))
    pdal_points = model.classify(pdal_points, "test")
    breakpoint()

#loops through all tiles in inDir
for tile in os.listdir(inDir):
    #discards .DS_store
    if (not tile.startswith('.')):
        #obtain name for file
        file = os.path.splitext(tile)[0]
        start=time.time()
        print(inDir+tile)

        #read file for points
        points, pdal_points = model.read(inDir+tile)
        
        #apply filters
        #pdal_points = model.applyFilters(pdal_points, [outlier])
        
        #performs segmentation on file
        cloud, non_ground, ground, pdal_points= model.smrf(pdal_points)
        labels, *_ = model.segment(points=cloud, image_path=outRaster+tile+"-raster.tif", labels_path=outLabels+tile+"-labeled.tif")
        points_grouped = model.grouping(pdal_points, labels, ground, non_ground)
        pdal_points = model.featureFilter(points_grouped, file)
        df = pd.DataFrame(pdal_points)
        table = pa.Table.from_pandas(df)
        ft.write_feather(table, 'arrows/points_filtered.feather')
        model.write(points=pdal_points, non_ground=non_ground, ground=ground, segment_ids=labels, save_path=outSeg+tile+"-segmented.las")
        end = time.time()
        print(f'Segment-lidar completed in {end - start:.2f} seconds.\n')
