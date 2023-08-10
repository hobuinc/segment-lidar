import re
import requests
import random
import sys
import os
import numpy as np
import pdal
import pyproj
from shapely.geometry import shape, Polygon, MultiPolygon
from shapely.ops import transform

pipelines = None
index_filename = 'lidar-index.geojson'


def get_extent(east, north, buffer):
    '''Obtains a bbox of length buffer around the points east and north
    '''

    point = {"type": "Point", "coordinates": [east, north]}
    point = shape(point)

    #UTM
    webmercator = pyproj.CRS("EPSG:3857")
    #Lat lng
    wgs84 = pyproj.CRS("EPSG:4326")

    #wgs_to_web = pyproj.Transformer.from_crs(wgs84, webmercator, always_xy=True).transform
    web_to_wgs = pyproj.Transformer.from_crs(webmercator, wgs84, always_xy=True).transform

    p = point
    #webmercator coordinates
    poly_wm = Polygon([(p.x - buffer, p.y - buffer), (p.x + buffer, p.y - buffer),
              (p.x + buffer, p.y + buffer), (p.x - buffer, p.y + buffer)])
    #wgs84
    poly_dd = transform(web_to_wgs, poly_wm)
    d = {}
    d['poly_dd'] = poly_dd
    d['poly_wm'] = poly_wm
    return d

def is_intersect(extent, location):
    ''' checks that the given extent intersects with the given location
    '''
    from osgeo import ogr
    minx, miny, maxx, maxy = extent['poly_dd'].bounds

    #check if there is an intersection with any of the polygons
    sql = f'SELECT url from "lidar-index" WHERE ST_CONTAINS(GEOMETRY, BuildMbr({minx:.8f}, {miny:.8f}, {maxx:.8f}, {maxy:.8f}))'
    l = ogr.Open(index_filename)
    r = np.array(l.ExecuteSQL(sql, dialect='SQLITE'))

    #Should only return true if there are points within the geometry
    if r is None or len(r) == 0:
        return False
    
    for url in r:
        if(url[0] == location):
            return True
    return False

#obtains bouding coordinates for the input location
def get_bounds(location):
    '''obtains the values from the bounds for the provided location
    '''
    bounds = np.asarray(requests.get(location).json().get('bounds'))
    return bounds

#creates pipelines for writing and filtering
# def get_pipelines(url, extent, options, name):

#     minx, miny, maxx, maxy = extent['poly_wm'].bounds

#     # PDAL's bounds format is kind of weird
#     bounds = f'([{minx:.8f}, {maxx:.8f}], [{miny:.8f}, {maxy:.8f}])'

#     reader = pdal.Reader.ept(url, bounds=bounds, requests=16)


#     ground_only = 'Classification[2:2]'
#     useful_classes = "Classification[0:6],Classification[17:17],Classification[9:9],Classification[10:10],Classification[11:11]"

#     dtm = reader | pdal.Filter.range(limits = ground_only) | \
#             pdal.Writer.gdal(filename=f'./data/dtm/{name}-dtm.tif',
#                              resolution=options['resolution'],
#                              dimension="Z",
#                              data_type = "float32",
#                              bounds=bounds,
#                              window_size=3)
#     intensity = reader |  \
#              pdal.Filter.assign(value = [ "Intensity = Intensity / 256"]) | \
#              pdal.Filter.range(limits = useful_classes) | \
#              pdal.Writer.gdal(f"./data/intensity/{name}-intensity.tif",
#                               resolution=options['resolution'],
#                               dimension="Intensity",
#                               bounds=bounds,
#                               data_type="uint8",
#                               output_type="mean")
#     dsm = reader| pdal.Filter.range(limits = useful_classes) | \
#              pdal.Writer.gdal(f"./data/dsm/{name}-dsm.tif",
#                               resolution=options['resolution'],
#                               dimension="Z",
#                               bounds=bounds,
#                               data_type = "float32",
#                               window_size = 3,
#                               output_type="idw")
#     hag = reader| pdal.Filter.hag_nn() | \
#              pdal.Writer.gdal(f"./data/hag/{name}-hag.tif",
#                               resolution=options['resolution'],
#                               dimension="HeightAboveGround",
#                               data_type = "float32",
#                               bounds=bounds,
#                               window_size = 3,
#                               output_type="idw")
#     numret = reader |  \
#              pdal.Filter.range(limits = useful_classes) | \
#              pdal.Writer.gdal(f"./data/numret/{name}-numret.tif",
#                               resolution=options['resolution'],
#                               dimension="NumberOfReturns",
#                               bounds=bounds,
#                               data_type="int16",
#                               output_type="idw")
#     classification = reader |  \
#              pdal.Writer.gdal(f"./data/class/{name}-classification.tif",
#                               resolution=options['resolution'],
#                               dimension="Classification",
#                               bounds=bounds,
#                               data_type="int16",
#                               output_type="idw")

#     pipelines = [dsm, intensity, dtm, hag, numret, classification]
#     # pipelines = [dsm, intensity]
#     return pipelines


def make_cloud(url, extent, file_num, name):
    '''creates a pipeline for turning obtained ept locations into pointclouds
    '''
    minx, miny, maxx, maxy = extent['poly_wm'].bounds

    # PDAL's bounds format is kind of weird
    bbox = f'([{minx:.8f}, {maxx:.8f}], [{miny:.8f}, {maxy:.8f}])'

    reader = pdal.Reader.ept(url, bounds=bbox, requests=16) | \
        pdal.Filter.assign(value = "Classification = 0")
    if use_seed:
        pointcloud = reader | \
                pdal.Writer.copc(f"./data/cloud/seed{seed}/{name}-{file_num}.copc.laz",
                                 forward="all")
    else:
        pointcloud = reader | \
                pdal.Writer.copc(f"./data/cloud/noseed/{name}-{file_num}.copc.laz",
                                 forward="all")
        
    return pointcloud


if __name__ == '__main__':  
    #make directories if they dont exist
    dirs = ["./data", "./data/cloud", "./data/cloud/noseed"]
    for i in dirs:
        if not os.path.exists(i):
            os.mkdir(i)

    rand = random
    file_num = 0

    #Edit these for different results
    num_locations = 300
    width = 300

    #Initialize seed for randomization and repetition
    use_seed = input('Would you like to use a seed? (y/n)').lower().strip()== 'y'
    
    #set seed
    if use_seed is True:
        seed = input("Please enter a seed value: ")
        rand.seed(seed)
        seed_path = f'./data/cloud/seed{seed}/'
        if not os.path.exists(seed_path):
            os.mkdir(seed_path)
            
    #obtain locations from USGS
    resources = 'https://usgs.entwine.io/boundaries/resources.geojson'
    features = requests.get(resources).json()['features']

    # obtain random locations
    locations = np.array([])
    for i in range(num_locations):
        rand_location = rand.randint(0, 2007)

        #obtain and save urls for USGS locations
        usgs_url = features[rand_location]['properties']['url']
        locations = np.append(locations, usgs_url)

    #sort locations for ease of use
    locations = np.sort(locations)

    #Save locations using seed value
    if use_seed is True:
        np.savetxt(f'{seed}-locations.txt', locations, fmt='%s', newline= '\n', header = f'Width: {width} \nSeed Value: {seed}')
        rand.seed(seed)
        print(f"Saved locations in seed{seed}-locations.txt")
    #Save locations if no seed value
    else:
        np.savetxt('noseed.txt', locations, fmt='%s', newline= '\n', header = f'Width: {width} \nNo Seed')
        print("Saved locations in noseed.txt")

    #read locations from generated file
    if use_seed:
        locations = np.loadtxt(f'{seed}-locations.txt', dtype = 'str')
    else:
        locations = np.loadtxt(f'noseed.txt', dtype = 'str')

    for loc in locations:
        #obtain location name
        name = re.search('https://s3-us-west-2.amazonaws.com/usgs-lidar-public/(.*)/ept.json', loc).group(1)
        bound = get_bounds(loc)
        
        fail_count = 0
        failed = False
        #search for bounding box that contains points
        INTERSECTION = False
        while not INTERSECTION:
            #Select random bounds
            north = rand.randint(bound[1], bound[4])
            east = rand.randint(bound[0], bound[3])
            #obtain extent
            extents = get_extent(east, north, width)
            
            # stops testing a file if cannot find bounds in a reasonable amnt of time
            INTERSECTION = is_intersect(extents, loc)
            if not INTERSECTION:
                fail_count += 1
            if fail_count == 15:
                failed = True
                break
        #moves on to next location if failure
        if failed:
            print("Failed", loc)
            file_num += 1
            continue

        if not INTERSECTION:
            print(f"Unable to fetch lidar data for bounds in '{loc}'")
            sys.exit()
        option = {"resolution":1.0,
               "filename":'out.tif',
               "ground_only": True}
        
        #create pipeline
        cloudpipe = make_cloud(loc, extents, file_num, name)
        file_num += 1
        with open('Pointcloud.json', 'wb') as j:
             j.write(cloudpipe.pipeline.encode('utf-8'))
             print("Wrote", loc)
        cloudpipe.execute()
        

        # for pipeline in pipelines:
        #     dimension = pipeline.stages[-1].options['dimension']
        #     #breakpoint()

            # with open(f'{dimension}.json','wb') as p:
            #     p.write(pipeline.pipeline.encode('utf-8'))
            # results = pipeline.execute()
    #       pipeline._del_executor()
    #       the last stage of our pipeline is the writer, and the 'dimension'
    #       option on the writer is what we want to print
    #       print (f"Number of points returned for dimension {dimension}: {results}")
            #pipelines, cloudpipe = get_pipelines(urls, extents, option, name)

    print("Finished making tiles. Located in [./data/cloud by default]")