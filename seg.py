import os
from posixpath import splitext
import pdal
import time
from segment_lidar import samlidar
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.feather as ft

model = samlidar.SamLidar(ckpt_path="seg-lidar/sam_vit_h_4b8939.pth")
inDir = "tiles/"
outRaster = "rasters/"
outLabels = "labels/"
outSeg = "segments/"
arrows = "arrows/"

li = [inDir, outRaster, outLabels, outSeg]
for i in li:
    try:
        os.path.exists(i)
    except:
        os.mkdir(i)

sort = pdal.Filter.sort(dimension="GPSTime")
outlier = pdal.Filter.outlier(method="radius", radius="1.0", min_k="4")
filters = [outlier, sort]


if 'points_filtered.feather' in os.listdir(arrows):
    feather = ft.read_table('arrows/points_filtered.feather')
    df1 = feather.to_pandas()
    s = df1.dtypes
    pdal_points = np.array([tuple(x) for x in df1.values], dtype=list(zip(s.index, s)))
    pdal_points = model.classify(pdal_points)
    breakpoint()
for tile in os.listdir(inDir):
    if (not tile.startswith('.')):
        breakpoint(print('no'))
        file = os.path.splitext(tile)[0]
        start=time.time()
        print(inDir+tile)
        points, pdal_points = model.read(inDir+tile)
        #pdal_points = model.applyFilters(pdal_points, [outlier])
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
