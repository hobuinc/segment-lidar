# Copyright (c) 2023 - University of Liège
# Author : Anass Yarroudh (ayarroudh@uliege.be), Geomatics Unit of ULiege
# This file is distributed under the BSD-3 licence. See LICENSE file for complete text of the license.

import os
import time
from typing import Dict, Optional, Set, cast
from matplotlib.pyplot import cla
from regex import F
import torch
import numpy as np
import CSF
from numpy.lib.recfunctions import structured_to_unstructured, unstructured_to_structured
from numpy.lib.recfunctions import append_fields
from numpy.lib import recfunctions as rfn
import supervision as sv
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from samgeo import SamGeo
from samgeo.text_sam import LangSAM
from typing import List, Tuple
import rasterio
import laspy
import pdal


# file formats supported
pdal_formats = set({".laz", ".las", ".ply", ".pcd", ".xyz", ".pts", ".txt", ".csv", ".asc", ".e57"})
npy_formats = set({".npy"})
formats = pdal_formats.copy()
formats.update(npy_formats)

# point cloud fields that needs to be casted/checked
position_fields = (
        ('X', 'Y', 'Z'), 
        ('x', 'y', 'z')
    )
position_fields_set = set(np.concatenate(position_fields).flat)
color_fields = (
    ('Red', 'Green', 'Blue'), 
    ('red', 'green', 'blue'), 
    ('red255', 'green255', 'blue255'), 
    ('r', 'g', 'b'), 
    ('R', 'G', 'B')
)
color_fields_set = set(np.concatenate(color_fields).flat)
intensity_fields = (
    ('Intensity'), ('intensity'), ('i'), ('I')
)
intensity_fields_set = set(intensity_fields)
classification_fields = (
    ('Classification'), ('classification'), ('c'), ('C')
)
classification_fields_set = set(classification_fields)


def cloud_to_image(points: np.ndarray, minx: float, maxx: float, miny: float, maxy: float, resolution: float) -> np.ndarray:
    """
    Converts a point cloud to an image.

    :param points: An array of points in the cloud, where each row represents a point.
                   The array shape can be (N, 3) or (N, 6).
                   If the shape is (N, 3), each point is assumed to have white color (255, 255, 255).
                   If the shape is (N, 6), the last three columns represent the RGB color values for each point.
    :type points: ndarray
    :param minx: The minimum x-coordinate value of the cloud bounding box.
    :type minx: float
    :param maxx: The maximum x-coordinate value of the cloud bounding box.
    :type maxx: float
    :param miny: The minimum y-coordinate value of the cloud bounding box.
    :type miny: float
    :param maxy: The maximum y-coordinate value of the cloud bounding box.
    :type maxy: float
    :param resolution: The resolution of the image in units per pixel.
    :type resolution: float
    :return: An image array representing the point cloud, where each pixel contains the RGB color values
             of the corresponding point in the cloud.
    :rtype: ndarray
    :raises ValueError: If the shape of the points array is not valid or if any parameter is invalid.
    """
    if points.shape[1] == 3:
        colors = np.array([255, 255, 255])
    else:
        colors = points[:, -3:]

    x = (points[:, 0] - minx) / resolution
    y = (maxy - points[:, 1]) / resolution
    pixel_x = np.floor(x).astype(int)
    pixel_y = np.floor(y).astype(int)

    width = int((maxx - minx) / resolution) + 1
    height = int((maxy - miny) / resolution) + 1

    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[pixel_y, pixel_x] = colors

    return image


def image_to_cloud(points: np.ndarray, minx: float, maxy: float, image: np.ndarray, resolution: float) -> List[int]:
    """
    Converts an image to a point cloud with segment IDs.

    :param points: An array of points representing the cloud, where each row represents a point.
                   The array shape is (N, 3) where each point contains x, y, and z coordinates.
    :param minx: The minimum x-coordinate value of the cloud bounding box.
    :param maxy: The maximum y-coordinate value of the cloud bounding box.
    :param image: The image array representing the input image.
    :param resolution: The resolution of the image in units per pixel.
    :return: A list of segment IDs for each point in the cloud. The segment ID represents the color segment
             in the input image that the corresponding point belongs to.
    :rtype: List[int]
    """
    segment_ids = []
    unique_values = {}
    image = np.asarray(image)

    # Calculate pixel coordinates for all points
    x = (points[:, 0] - minx) / resolution
    y = (maxy - points[:, 1]) / resolution
    pixel_x = np.floor(x).astype(int)
    pixel_y = np.floor(y).astype(int)

    # Mask points outside image bounds
    out_of_bounds = (pixel_x < 0) | (pixel_x >= image.shape[1]) | (pixel_y < 0) | (pixel_y >= image.shape[0])
    segment_ids.extend([-1] * np.sum(out_of_bounds))

    # Extract RGB values for valid points
    valid_points = ~out_of_bounds
    rgb = image[pixel_y[valid_points], pixel_x[valid_points]]

    # Map RGB values to unique segment IDs
    for rgb_val in rgb:
        if rgb_val not in unique_values:
            unique_values[rgb_val] = len(unique_values)

        segment_ids.append(unique_values[rgb_val])

    return segment_ids


class mask:
    def __init__(self, crop_n_layers: int = 1, crop_n_points_downscale_factor: int = 1, min_mask_region_area: int = 1000, points_per_side: int = 32, pred_iou_thresh: float = 0.80, stability_score_thresh: float = 0.8):
        """
        Initializes an instance of the mask class.

        :param crop_n_layers: The number of layers to crop from the top of the image, defaults to 1.
        :type crop_n_layers: int
        :param crop_n_points_downscale_factor: The downscale factor for the number of points in the cropped image, defaults to 1.
        :type crop_n_points_downscale_factor: int
        :param min_mask_region_area: The minimum area of a mask region, defaults to 1000.
        :type min_mask_region_area: int
        :param points_per_side: The number of points per side of the mask region, defaults to 32.
        :type points_per_side: int
        :param pred_iou_thresh: The IoU threshold for the predicted mask region, defaults to 0.90.
        :type pred_iou_thresh: float
        :param stability_score_thresh: The stability score threshold for the predicted mask region, defaults to 0.92.
        :type stability_score_thresh: float
        """
        self.crop_n_layers = crop_n_layers
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area
        self.points_per_side = points_per_side
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh


class SamLidar:
    def __init__(self, ckpt_path: str, algorithm: str, model_type: str = 'vit_h', resolution: float = 0.25, device: str = 'cuda:0', sam_kwargs: bool = False) -> None:
        """
        Initializes an instance of the SamLidar class.

        :param algorithm: The algorithm to use, defaults to 'segment-geospatial'.
        :type algorithm: str
        :param model_type: The type of the model, defaults to 'vit_h'.
        :type model_type: str
        :param ckpt_path: The path to the model checkpoint.
        :type ckpt_path: str
        :param resolution: The resolution value, defaults to 0.25.
        :type resolution: float
        :param device: The device to use, defaults to 'cuda:0'.
        :type device: str
        :param sam_kwargs: Whether to use the SAM kwargs when using 'segment-geospatial' as algorithm, defaults to False.
        :type sam_kwargs: bool
        """
        self.algorithm = algorithm
        self.model_type = model_type
        self.ckpt_path = ckpt_path
        self.resolution = resolution
        self.device = torch.device('cuda:0') if device == 'cuda:0' and torch.cuda.is_available() else torch.device('cpu')
        self.mask = mask()

        if sam_kwargs or algorithm == 'segment-anything':
            self.mask_generator = SamAutomaticMaskGenerator(
                model=sam_model_registry[model_type](checkpoint=ckpt_path).to(device=self.device),
                crop_n_layers=self.mask.crop_n_layers,
                crop_n_points_downscale_factor=self.mask.crop_n_points_downscale_factor,
                min_mask_region_area=self.mask.min_mask_region_area,
                points_per_side=self.mask.points_per_side,
                pred_iou_thresh=self.mask.pred_iou_thresh,
                stability_score_thresh=self.mask.stability_score_thresh
            )

        if sam_kwargs:
            self.sam_kwargs = {
                'crop_n_layers': self.mask.crop_n_layers,
                'crop_n_points_downscale_factor': self.mask.crop_n_points_downscale_factor,
                'min_mask_region_area': self.mask.min_mask_region_area,
                'points_per_side': self.mask.points_per_side,
                'pred_iou_thresh': self.mask.pred_iou_thresh,
                'stability_score_thresh': self.mask.stability_score_thresh
            }
        else:
            self.sam_kwargs = None

        self.sam_geo = SamGeo(
            model_type=self.model_type,
            checkpoint=self.ckpt_path,
            device=self.device,
            sam_kwargs=self.sam_kwargs
        )

        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'




    def read(self, path: str, classification: Optional[int or list[int] or Tuple[int, ...]] = None) -> np.ndarray:

        """
        Reads a point cloud from a file and returns it as a NumPy array.

        :param path: The path to the input file.
        :type path: str
        :param classification: The optional classification value to filter the point cloud, defaults to None.
        :type classification: int, optional
        :return: The point cloud as a NumPy array.
        :rtype: np.ndarray
        :raises ValueError: If the input file format is not supported.
        """
        
        start = time.time()
        extension = os.path.splitext(path)[1]
        
        if extension not in formats:
            try:
                raise ValueError(f'The input file format {extension} is not supported.\nThe file format should be one of {formats}.')
            except ValueError as error:
                message = str(error)
                lines = message.split('\n')
                print(lines[-2])
                print(lines[-1])
                exit()

        print(f'Reading {path}...')

        pcd: np.ndarray
        fields: Tuple[str,...] = ()
        fields_set: Set[str] = set()
        
        if extension in npy_formats:
            pcd = np.load(path)
        else:
            pipeline = pdal.Pipeline(f"[\"{path}\"]")
            count = pipeline.execute()
            pcd = pipeline.arrays[0]
        
        dtypes = pcd.dtype
        
        fields = tuple(dtypes.names)
        fields_set = set(fields)
        has_position = any(set(c).issubset(fields_set) for c in position_fields)
        has_color = any(set(c).issubset(fields_set) for c in color_fields)
        has_intensity = any(set(c).issubset(fields_set) for c in intensity_fields)
        has_classification = any(set(c).issubset(fields_set) for c in classification_fields)
        if not has_position:
            raise ValueError(f'The input file does not contain position values.')
        if not has_classification and classification is not None:
            raise ValueError(f'The input file does not contain classification values.')
        else:
            for label in fields:
                if classification is not None and label in classification_fields_set:
                    for c in cast(tuple[int], classification):
                        pcd = pcd[pcd[label] == c]
        if isinstance(classification, int):
            classification = (classification,)
        elif isinstance(classification, list):
            classification = tuple(classification)

        dtypes_casted = np.dtype([(label, np.float64) if label in position_fields_set or label in color_fields_set or label in intensity_fields_set else (label, dtypes[label]) for label in fields])
        pcd_casted = pcd.astype(dtypes_casted, copy=True)
        
        for label in fields:
            if label in color_fields_set or label in intensity_fields_set:
                if dtypes[label] == np.uint8:
                    pcd_casted[label] /= 255.0
                elif dtypes[label] == np.uint16:
                    pcd_casted[label] /= 65535.0
       
        end = time.time()
        print(f'File reading is completed in {end - start:.2f} seconds. The point cloud contains {pcd_casted.shape[0]} points.\n')

        return (pcd_casted)
  
    def applyFilters(self, pdal_points: np.ndarray, filters=[], multi_array=False) -> np.ndarray:
        """
        Gives ability to apply PDAL Filters before segmentation such as denoising.


        :param pdal_points: The input point cloud as a structured array.

        :type pdal_points: np.ndarray
        :param filters: Array of the PDAL filters defined to be applied.
        :type filters: array
        :return: The input point cloud as a NumPy array after filters were applied.
        :rtype: np.ndarray
        """
        pipeline = pdal.Pipeline(arrays=[pdal_points])
        for f in filters:
            pipeline = pipeline | f

        count = pipeline.execute()
        if multi_array==False:
            if len(pipeline.arrays) > 1:
                raise Exception("the array count was not 1!")
            pdal_points = pipeline.arrays[0]
        else:
            pdal_points = pipeline.arrays
        return pdal_points
    
    def noiseFilter(self, pdal_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filters out noise points with PDAL filters.
    
        :param pdal_points: The input point cloud as a NumPy structured array.
        :type pdal_points: np.ndarray
        :return: A tuple containing a NumPy array with noise points removed, and a 
        NumPy array with only noise points.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        start = time.time()
        print(f'Applying PDAL noise filter ...')
        outlier = pdal.Filter.outlier(method="statistical", multiplier="3", mean_k="8")
        filtered = SamLidar.applyFilters(self, pdal_points, [outlier])
        noise = np.where(filtered['Classification']==7) or np.where(filtered['Classification'] > 20)
        noise_pts = filtered[noise]
        pdal_points = np.delete(filtered, noise)
        end = time.time()
        print(f'Noise filtering is completed in {end - start:.2f} seconds. The filtered cloud contains {pdal_points.shape[0]} points.\n')
        return pdal_points, noise_pts

         
    def csf(self, points: np.ndarray, class_threshold: float = 0.5, cloth_resolution: float = 0.2, iterations: int = 500, slope_smooth: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Applies the CSF (Cloth Simulation Filter) algorithm to filter ground points in a point cloud.

        :param points: The input point cloud as a NumPy array, where each row represents a point with x, y, z coordinates.
        :type points: np.ndarray
        :param class_threshold: The threshold value for classifying points as ground/non-ground, defaults to 0.5.
        :type class_threshold: float, optional
        :param cloth_resolution: The resolution value for cloth simulation, defaults to 0.2.
        :type cloth_resolution: float, optional
        :param iterations: The number of iterations for the CSF algorithm, defaults to 500.
        :type iterations: int, optional
        :param slope_smooth: A boolean indicating whether to enable slope smoothing, defaults to False.
        :type slope_smooth: bool, optional
        :return: A tuple containing three arrays: the filtered point cloud, non-ground (filtered) points indinces and ground points indices.
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
        """
        start = time.time()
        print(f'Applying CSF algorithm...')
        csf = CSF.CSF()
        csf.params.bSloopSmooth = slope_smooth
        csf.params.cloth_resolution = cloth_resolution
        csf.params.interations = iterations
        csf.params.class_threshold = class_threshold
        csf.setPointCloud(points[:, :3])
        ground = CSF.VecInt()
        non_ground = CSF.VecInt()
        csf.do_filtering(ground, non_ground)
        print(points)
        non_ground = np.asarray(non_ground)
        ground = np.asarray(ground)
        print("ground:",ground)
        np.info(ground)
        print("non ground:",non_ground)
        np.info(non_ground)
        

        points = points[non_ground, :]
        os.remove('cloth_nodes.txt')

        end = time.time()
        print(f'CSF algorithm is completed in {end - start:.2f} seconds. The filtered non-ground cloud contains {points.shape[0]} points.\n')

        return points, np.asarray(non_ground), np.asarray(ground)
    
   
    def smrf(self, pdal_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Applies the SMRF (Simple Morphological Filter) algorithm to filter ground points in a point cloud.
        Also converts the non-ground point cloud from a structured array to an unstructured XYZ-only array.
        :param pdal_points: The input point cloud as a NumPy structured array.
        :type points: np.ndarray
        :return: A tuple containing three arrays: the filtered point cloud, non-ground (filtered) points indinces and ground points indices.
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
        """
        start = time.time()
        print(f'Applying SMRF algorithm...')
        smrf = pdal.Filter.smrf()
        reclass = pdal.Filter.assign(value='Classification = 0') | \
                 pdal.Filter.assign(value="ReturnNumber = 1 where ReturnNumber < 1") | \
                 pdal.Filter.assign(value="NumberOfReturns = 1 where NumberOfReturns < 1")

        filter_list = [reclass, smrf]
        pdal_points = SamLidar.applyFilters(self, pdal_points, filter_list)

        ground = np.where(pdal_points["Classification"]==2)
        non_ground = np.where(pdal_points["Classification"]!=2)
        x = pdal_points["X"]
        y = pdal_points["Y"]
        z = pdal_points["Z"]
        points = np.vstack((x, y, z)).T
        points = np.delete(points, ground, 0)
        ground = np.ravel(ground)
        non_ground = np.ravel(non_ground)
        end = time.time()

        print(f'SMRF algorithm is completed in {end - start:.2f} seconds. The filtered non-ground cloud contains {points.shape[0]} points.\n')

        return points, non_ground, ground, pdal_points 

    def segment(self, points: np.ndarray, text_prompt: str = None, image_path: str = 'raster.tif', labels_path: str = 'labeled.tif') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Segments a point cloud based on the provided parameters and returns the segment IDs, original image, and segmented image.

        :param points: The point cloud data as a xyz NumPy array.
        :type points: np.ndarray
        :param text_prompt: Optional text prompt for segment generation, defaults to None.
        :type text_prompt: str
        :param image_path: Path to the input raster image, defaults to 'raster.tif'.
        :type image_path: str
        :param labels_path: Path to save the labeled output image, defaults to 'labeled.tif'.
        :type labels_path: str
        :return: A tuple containing the segment IDs, segmented image, and RGB image.
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
        """
        start = time.time()
        print(f'Segmenting the point cloud...')
        minx = np.min(points[:, 0])
        maxx = np.max(points[:, 0])
        miny = np.min(points[:, 1])
        maxy = np.max(points[:, 1])

        print(f'- Generating raster image...')
        image = cloud_to_image(points, minx, maxx, miny, maxy, self.resolution)
        image = np.asarray(image).astype(np.uint8)

        print(f'- Saving raster image...')
        with rasterio.open(
            image_path,
            'w',
            driver='GTiff',
            width=image.shape[1],
            height=image.shape[0],
            count=3,
            dtype=image.dtype
        ) as dst:
            for i in range(3):
                dst.write(image[:, :, i], i + 1)

        with rasterio.open(image_path, 'r') as src:
            image_rgb = src.read()

        print(f'- Applying {self.algorithm} to raster image...')
        if self.algorithm == 'segment-anything':
            sam = sam_model_registry[self.model_type](checkpoint=self.ckpt_path)
            sam.to(self.device)

            image_rgb = image_rgb.transpose(1, 2, 0)
            result = self.mask_generator.generate(image_rgb)
            mask_annotator = sv.MaskAnnotator()
            detections = sv.Detections.from_sam(result)
            num_masks, height, width = detections.mask.shape
            segmented_image = np.zeros((height, width), dtype=np.uint8)
            for i in range(num_masks):
                mask = detections.mask[i]
                segmented_image[mask] = i

            print(f'- Saving segmented image...')
            with rasterio.open(
                labels_path,
                'w',
                driver='GTiff',
                width=segmented_image.shape[1],
                height=segmented_image.shape[0],
                count=1,
                dtype=segmented_image.dtype
            ) as dst:
                dst.write(segmented_image, 1)


        elif self.algorithm == 'segment-geospatial':
            if text_prompt is not None:
                print(f'- Generating labels using text prompt...')
                sam = LangSAM()
                sam.predict(image=image_path, text_prompt=text_prompt, box_threshold=0.3, text_threshold=0.3, output=labels_path)
                print(f'- Saving segmented image...')
            else:
                sam = self.sam_geo
                sam.generate(source=image_path, output=labels_path, erosion_kernel=(3,3), foreground=True, unique=True)
                print(f'- Saving segmented image...')

        with rasterio.open(labels_path, 'r') as src:
            segmented_image = src.read()
            segmented_image = np.squeeze(segmented_image)

        print(f'- Generating segment IDs...')
        segment_ids = image_to_cloud(points, minx, maxy, segmented_image, self.resolution)
        end = time.time()

        print(f'Segmentation is completed in {end - start:.2f} seconds. Number of instances: {np.max(segmented_image)}\n')
        return segment_ids, segmented_image, image_rgb
    
    def grouping(self, pdal_points: np.ndarray, segment_ids: np.ndarray, ground, non_ground, noise: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Creates a list of arrays with the segments grouped by their segment_id; merges noise points (optional)
    
        :param pdal_points: The point cloud data as a structured NumPy array.
        :type pdal_points: np.ndarray
        :param segment_ids: NumPy array of the segments.
        :type segment_ids: np.ndarray
        :param ground: array with the indices of all ground points
        :type ground: array
        :param non_ground: array with the indices of all non-ground points
        :type non_ground: array
        :param noise: NumPy array with all points classified as noise from the noiseFilter function
        :type noise: np.ndarray, optional
        :return: Numpy array that has points grouped by their segment_id.
        :rtype: np.ndarray 
        """
        start = time.time()
        print('Grouping points by segment ID...')
        if ground is not None:
            segment_ids = np.append(segment_ids, np.full(len(ground), -1))
            indices = np.append(non_ground, ground)
        else:
            indices = non_ground
        points_ordered = np.take(pdal_points, indices)
        if noise is not None:
            noise_types = noise.dtype.descr
            fields = []
            for field in points_ordered.dtype.descr:
                if field not in noise_types:
                    fields.append(field)
            if len(fields) != 0: 
                temp = np.full((len(fields), noise.shape[0]), 0, dtype=fields)
                noise = rfn.merge_arrays((noise, temp), flatten=True)
            points_ordered = np.append(points_ordered, noise)
            segment_ids = np.append(segment_ids, np.full(len(noise), -2))

        temp = np.full(points_ordered.shape[0], 1, dtype=[('segment_id', 'i4')])
        merged = rfn.merge_arrays((points_ordered, temp), flatten=True)
        merged['segment_id'] = segment_ids
        groupby = pdal.Filter.groupby(dimension="segment_id", where = "segment_id >= 0")
        filters = [groupby]
        
        points_grouped = SamLidar.applyFilters(self, merged, filters, multi_array=True)
        end = time.time()

        print(f'Grouping is complete in {end - start:.2f} seconds.')  
        return points_grouped
    
    def featureFilter(self, points_grouped: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        """
        Adds PDAL Dimesionality Data Types to the Segments and merges the list of arrays.
    
        :param points_grouped: The input point cloud data as an array of multiple structured NumPy arrays.
        :type points: np.ndarray
        :return: A tuple containing a NumPy array with average PDAL Dimesionality Data Types of each segment,
        and a NumPy array containing ground, non-segmented points, and small segments.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        start = time.time()
        print("Applying PDAL 'covariancefeatures' filters based on segment ID")
        bad_pts = points_grouped[0]
        points_grouped = points_grouped[1:]
        if len(points_grouped) > 1:

            cov = pdal.Filter.covariancefeatures(feature_set="Dimensionality")

            filters = [cov]
            points_filtered = []
            small_pts = []

            features = ['Linearity', 'Planarity', 'Scattering', 'Verticality']

            for array in points_grouped:
                if len(array) >= 10:
                    filtered = SamLidar.applyFilters(self, array, filters)
                    for col in features:
                        filtered[col] = np.full(len(filtered), np.median(filtered[col]))
                    temp = np.full(filtered.shape[0], np.mean(filtered['NumberOfReturns']), dtype=[('meanNumReturns', '<f8')])
                    filtered = rfn.merge_arrays((filtered, temp), flatten =True)
                    points_filtered.append(filtered)
                else:
                    small_pts.append(array)       
            points_filtered = np.concatenate(points_filtered)
            if small_pts:
                small_pts = np.concatenate(small_pts)
                bad_pts = np.append(bad_pts, small_pts)
        else:
            points_filtered = None
            print("No segments to be filtered")
        end = time.time()
        print(f'Filtering is complete in {end - start:.2f} seconds.')  
        return points_filtered, bad_pts

    def classify(self, points_filtered, bad_pts, name, location, output_txt: bool = False):
        """

        Classifies the Point Cloud Data using the Dimesionality Data Types.

        :param points_filtered: The input point cloud data as a NumPy array after getting Dimensionality Data Types.
        :type points_filtered: np.ndarray
        :param file: Name of input file without extension
        :type file: string
        :return: points_filtered
        :rtype: np.ndarray
        """ 

         #obtain one of each segment ID for percentile info 
        uniq_id, indicies = np.unique(points_filtered[:]['segment_id'], return_index = True)
        uniq_id = points_filtered[indicies]
        uniq_groundless = np.delete(arr = uniq_id, axis=0, obj = (np.where(uniq_id['Classification'] == 2))) 

        max_scatter = uniq_groundless['Scattering'].max()
        max_planar = uniq_groundless['Planarity'].max()
        max_mean_ret = uniq_groundless['meanNumReturns'].max() 
        max_linearity = uniq_groundless['Linearity'].max()
        points_filtered['Scattering'] = points_filtered['Scattering']/ max_scatter
        points_filtered['Planarity'] = points_filtered['Planarity']/ max_planar
        points_filtered['meanNumReturns'] = points_filtered['meanNumReturns']/ max_mean_ret
        points_filtered['Linearity'] = points_filtered['Linearity']/ max_linearity

        # #obtain percentiles 
        scatter_perc = np.percentile(a = uniq_groundless['Scattering'], q = np.arange(10, 100, 10)) 
        planar_perc = np.percentile(a = uniq_groundless['Planarity'], q = np.arange(10, 100, 10)) 
        mean_ret_perc = np.percentile(a = uniq_groundless['meanNumReturns'], q = np.arange(10, 100, 10)) 
        linearity_perc = np.percentile(a = uniq_groundless['Linearity'], q = np.arange(10, 100, 10)) 

        #Save percentile outputs
        if output_txt == True:
            np.savetxt(f'./data/class/{location}/scattering/{name}-scatter.txt', scatter_perc)
            np.savetxt(f'./data/class/{location}/planarity/{name}-planarity.txt', planar_perc)
            np.savetxt(f'./data/class/{location}/meanRet/{name}-meanRet.txt', mean_ret_perc)
            np.savetxt(f'./data/class/{location}/linearity/{name}-linearity.txt', linearity_perc)

        veg = np.where(points_filtered['Scattering'] >= .5) and np.where(points_filtered['Classification'] != 2)
        build = np.where(points_filtered['Classification'] != 2) and np.where(points_filtered['Planarity'] > .5) and np.where(points_filtered['Scattering'] < .5)


        classes = [veg, build]
        for obj in classes:
            obj = np.array(obj)
            obj = np.ravel(obj)

        np.put(points_filtered['Classification'], veg, 4)
        np.put(points_filtered['Classification'], build, 6)


        writer_options = {'forward': 'all'}
        p = getattr(pdal.Writer, 'copc')(filename = f'./data/class/{location}/all-info/{name}.copc.laz', **writer_options).pipeline(points_filtered)
        p.execute()

        base_fields = list(bad_pts.dtype.names)
        points_filtered = points_filtered[base_fields]
        classified_points = np.append(points_filtered, bad_pts)

        return classified_points



    def write(self, points: np.ndarray, segment_ids: np.ndarray, non_ground: np.ndarray = None, ground: np.ndarray = None, save_path: str = 'segmented.las') -> None:
        """
        Writes the segmented point cloud data to a LAS/LAZ file.

        :param points: The input point cloud data as a NumPy array, where each row represents a point with x, y, z coordinates.
        :type points: np.ndarray
        :param segment_ids: The segment IDs corresponding to each point in the point cloud.
        :type segment_ids: np.ndarray
        :param non_ground: Optional array of indices for non-ground points in the original point cloud (default: None).
        :type non_ground: np.ndarray, optional
        :param ground: Optional array of indices for ground points in the original point cloud (default: None).
        :type ground: np.ndarray, optional
        :param save_path: The path to save the segmented LAS/LAZ file (default: 'segmented.las').
        :type save_path: str, optional
        :return: None
        """
        start = time.time()
        extension = os.path.splitext(save_path)[1]
        try:
            if extension not in ['.laz', '.las']:
                raise ValueError(f'The input file format {extension} is not supported.\nThe file format should be [.las/.laz].')
        except ValueError as error:
            message = str(error)
            lines = message.split('\n')
            print(lines[-2])
            print(lines[-1])
            exit()

        print(f'Writing the segmented point cloud to {save_path}...')

        header = laspy.LasHeader(point_format=3, version="1.3")
        lidar = laspy.LasData(header=header)

        if ground is not None:
            indices = np.append(non_ground, ground)
            lidar.xyz = points[indices]
            segment_ids = np.append(segment_ids, np.full(len(ground), -1))
            lidar.add_extra_dim(laspy.ExtraBytesParams(name="segment_id", type=np.int32))
            lidar.segment_id = segment_ids
        else:
            lidar.xyz = points
            lidar.add_extra_dim(laspy.ExtraBytesParams(name="segment_id", type=np.int32))
            lidar.segment_id = segment_ids

        lidar.write(save_path) 

        end = time.time()
        print(f'Writing is completed in {end - start:.2f} seconds.\n')
        return None
    
    

    def write_pdal(self, points: np.ndarray, segment_ids: np.ndarray, non_ground: Optional[np.ndarray] = None, ground: Optional[np.ndarray] = None, save_path: str = 'segmented.laz', dtypes: Optional[np.dtype] = None):

        """
        Writes the segmented point cloud data to any pointcloud format supported by PDAL, checks if the array is structured or not, if it is not a structured array it will convert to structured array.

        :param points: The input point cloud data as a NumPy array whether structured or not.
        :type points: np.ndarray
        :param segment_ids: The segment IDs corresponding to each point in the point cloud.
        :type segment_ids: np.ndarray
        :param non_ground: Optional array of indices for non-ground points in the original point cloud (default: None).
        :type non_ground: np.ndarray, optional
        :param ground: Optional array of indices for ground points in the original point cloud (default: None).
        :type ground: np.ndarray, optional
        :param save_path: The path to save the segmented LAS/LAZ file (default: 'segmented.laz').
        :type save_path: str, optional
        :return: None
        """
        start = time.time()
        
        extension = os.path.splitext(save_path)[1]
        try:
            if extension not in formats:
               raise ValueError(f'The input file format {extension} is not supported.\nThe file format should be [.las/.laz].')
        except ValueError as error:
            message = str(error)
            lines = message.split('\n')
            print(lines[-2])
            print(lines[-1])
            exit()

        print(f'Writing the segmented point cloud to {save_path}...')
        if points.dtype.names is None:

            indices = np.arange(len(points))
        
            if ground is not None and non_ground is not None:
                indices = np.concatenate((non_ground, ground))
                data = points[indices]
                segment_ids = np.append(segment_ids, np.full(len(ground), -1))
            else:
                data = points[indices]
        
            data = np.hstack((data, segment_ids.reshape(-1, 1)))

            if dtypes is None or dtypes.names is None or len(dtypes.names) == 0:
                if data.shape[0] == 4:
                    dtypes = np.dtype([('X', np.float64), ('Y', np.float64), ('Z', np.float64), ('segment_id', np.int32)])
                elif data.shape[0] == 7:
                    dtypes = np.dtype([('X', np.float64), ('Y', np.float64), ('Z', np.float64), ('Red', np.uint16), ('Green', np.uint16), ('Blue', np.uint16), ('segment_id', np.int32)])
                elif data.shape[0] == 24:
                        dtypes = np.dtype([('X', np.float64), ('Y', np.float64), ('Z', np.float64), ('Red', np.uint16), ('Green', np.uint16), ('Blue', np.uint16), ('Intensity', np.uint16), ('ReturnNumber', np.uint8), ('NumberOfReturns', np.uint8), ('ScanDirectionFlag', np.uint8), ('EdgeOfFlightLine', np.uint8), ('Classification', np.uint8), ('ScanAngleRank', np.float64), ('UserData', np.uint8), ('PointSourceId', np.uint16), ('GpsTime', np.float64), ('segment_id', np.int32), ('Linearity', np.float64), ('Planarity', np.float64), ('Scattering', np.float64), ('Verticality', np.float64), ('Eigenvalue0', np.float64), ('Eigenvalue1', np.float64), ('Eigenvalue2', np.float64)])

                    #('X', np.float64), ('Y', np.float64), ('Z', np.float64), ('Intensity', '<u2'), ('ReturnNumber', 'u1'), ('NumberOfReturns', 'u1'), ('ScanDirectionFlag', 'u1'), ('EdgeOfFlightLine', 'u1'), ('Classification', 'u1'), ('ScanAngleRank', '<f4'), ('UserData', 'u1'), ('PointSourceId', '<u2'), ('GpsTime', np.float64), ('Red', '<u2'), ('Green', '<u2'), ('Blue', '<u2'), ('seg_id', '<i4'), ('Linearity', np.float64), ('Planarity', np.float64), ('Scattering', np.float64), ('Verticality', np.float64), ('Eigenvalue0', np.float64), ('Eigenvalue1', np.float64), ('Eigenvalue2', np.float64)]

                
                else:
                    dtype_descriptions = [('X', np.float64), ('Y', np.float64), ('Z', np.float64), ('Red', np.uint16), ('Green', np.uint16), ('Blue', np.uint16)]
                    dtype_descriptions.extend([(f'scalar_{i}', np.float64 if isinstance(i, float) else (np.int64 if isinstance(i, int) else str)) for i in range(data.shape[1] - 7)])
                    dtype_descriptions.append(('segment_id', np.int32))
                    dtypes = np.dtype(dtype_descriptions)
            dtypes_casted = np.dtype([(label, np.float64) if label in position_fields_set or label in color_fields_set or label in intensity_fields_set else (label, dtypes[label]) for label in dtypes.names]) # type: ignore
        
        
            pcd = unstructured_to_structured(points[indices], dtypes_casted)

            for label in dtypes.names: # type: ignore
                if label in color_fields_set:
                    print(f'- Adding color:{label} to the segmented point cloud...')
                    if label[0].lower() == 'r':
                        pcd['Red'] = (pcd[label] * 65536.0).astype(np.uint16)
                    elif label[0].lower() == 'g':
                        pcd['Green'] = (pcd[label] * 65536.0).astype(np.uint16)
                    else:
                        pcd['Blue'] = (pcd[label] * 65536.0).astype(np.uint16)
                elif label not in position_fields_set:
                    pcd[label] = pcd[label]
                    pcd = pcd.astype(dtypes, copy=True)
                    
        else:
            if 'segment_id' not in points.dtype.names:
                if ground is not None:
                    segment_ids = np.append(segment_ids, np.full(len(ground), -1))
                    indices = np.append(non_ground, ground)
                else:
                    indices = non_ground
                points_ordered = np.take(points, indices)
                temp = np.full(points_ordered.shape[0], 1, dtype=[('segment_id', 'i4')])
                points = rfn.merge_arrays((points_ordered, temp), flatten=True)
                points['segment_id'] = segment_ids

            pcd = points 
        writer_name = "las"
        writer_options = {'extra_dims': 'all', 'compression': 'laszip'}
        if extension == '.las':
            writer_name = 'las'
            writer_options = {'extra_dims': 'all'}
        elif extension in ['.xyz','.txt','.csv','.asc','.pts']:
            writer_name = 'text'
            writer_options = {'order': dtypes.names.join(','), 'precision': 14} # type: ignore
        elif extension == '.ply':
            writer_name = 'ply'
            writer_options = {'storage_mode': 'little_endian'}
        elif extension == '.e57':
            writer_name = 'e57'
        elif extension == '.pcd':
            writer_name = 'pcd'
            writer_options = {'compression': 'compressed', 'order': dtypes.names.join(','), 'precision': 14} # type: ignore
        elif '.copc.laz' in save_path:
            writer_name = 'copc'
            writer_options = {'forward': 'all'}   
        p = getattr(pdal.Writer, writer_name)(filename=save_path, **writer_options).pipeline(pcd)

        r = p.execute()

        
        end = time.time()
        print(f'Writing is completed in {end - start:.2f} seconds.\n')
