import os
import pdal
from segment_lidar import samlidar

model = samlidar.SamLidar(ckpt_path="seg-lidar/sam_vit_h_4b8939.pth")
inDir = "tiles/"
outRaster = "rasters/"
outLabels = "labels/"
outSeg = "segments/"

sort = pdal.Filter.sort(dimension="GPSTime")
outlier = pdal.Filter.outlier(method="radius", radius="1.0", min_k="4")
filters = [outlier, sort]

for tile in os.listdir(inDir):
    print(inDir+tile)
    points, pdal_points = model.read(inDir+tile)
    #pdal_points = model.applyFilters(pdal_points, [outlier])
    #cloud, non_ground, ground = model.csf(points)
    cloud, non_ground, ground = model.smrf(pdal_points)
    breakpoint()
    labels, *_ = model.segment(points=cloud, image_path=outRaster+tile+"-raster.tif", labels_path=outLabels+tile+"-labeled.tif")
    model.write(points=points, non_ground=non_ground, ground=ground, segment_ids=labels, save_path=outSeg+tile+"-segmented.las")