import os
from posixpath import splitext
import pdal
import time
from segment_lidar import samlidar


model = samlidar.SamLidar(ckpt_path="seg-lidar/sam_vit_h_4b8939.pth")
inDir = "tiles/"
outRaster = "rasters/"
outLabels = "labels/"
outSeg = "segments/"
outGroup = "groups/"

sort = pdal.Filter.sort(dimension="GPSTime")
outlier = pdal.Filter.outlier(method="radius", radius="1.0", min_k="4")
filters = [outlier, sort]

for tile in os.listdir(inDir):
    if (not tile.startswith('.')):
        file = os.path.splitext(tile)[0]
        start=time.time()
        print(inDir+tile)
        dtypes, points, pdal_points = model.read(inDir+tile)
        pdal_points = model.applyFilters(pdal_points, [outlier])
        cloud, non_ground, ground, pdal_points= model.smrf(pdal_points)
        labels, *_ = model.segment(points=cloud, image_path=outRaster+tile+"-raster.tif", labels_path=outLabels+tile+"-labeled.tif")
        points_grouped = model.grouping(pdal_points, labels, ground, non_ground)
        pdal_points = model.featureFilter(points_grouped, file)
        model.write_pdal(points=points, non_ground=non_ground, ground=ground, segment_ids=labels, save_path=outSeg+tile+"-segmented.las")
        end = time.time()
        print(f'Segment-lidar completed in {end - start:.2f} seconds.\n')