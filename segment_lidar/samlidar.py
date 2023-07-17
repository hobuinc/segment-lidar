# Copyright (c) 2023 - University of Liège
# Author : Anass Yarroudh (ayarroudh@uliege.be), Geomatics Unit of ULiege
# This file is distributed under the BSD-3 licence. See LICENSE file for complete text of the license.

import os
import time
import torch
from typing import Dict, Optional, Set, cast
from matplotlib.pyplot import cla
from regex import F
import torch
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
from numpy.lib.recfunctions import unstructured_to_structured
from numpy.lib import recfunctions as rfn
import supervision as sv
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from samgeo import SamGeo
from samgeo.text_sam import LangSAM
from typing import List, Tuple
import rasterio
import laspy
import pdal
import cv2

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
    def __init__(self, crop_n_layers: int = 1, crop_n_points_downscale_factor: int = 1, min_mask_region_area: int = 200, points_per_side: int = 5, pred_iou_thresh: float = 0.90, stability_score_thresh: float = 0.92):
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
    def __init__(self, ckpt_path: str, algorithm: str = 'segment-geospatial', model_type: str = 'vit_h', resolution: float = 0.25, device: str = 'cuda:0', sam_kwargs: bool = False) -> None:
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



    def read(self, path: str, classification: int = None) -> Tuple[np.ndarray, np.ndarray]:
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
        try:
            if extension not in ['.laz', '.las', '.npy']:
                raise ValueError(f'The input file format {extension} is not supported.\nThe file format should be [.las/.laz].')
        except ValueError as error:
            message = str(error)
            lines = message.split('\n')
            print(lines[-2])
            print(lines[-1])
            exit()

        print(f'Reading {path}...')

        if extension == '.npy':
            points = np.load(path)
        elif extension in ['.laz', '.las']:
            reader = pdal.Reader.las(filename=path)
            pipeline = reader.pipeline()
        
            count = pipeline.execute()
            pdal_points = pipeline.arrays[0]

            las = laspy.read(path)
            if classification == None:
                print(f'- Classification value is not provided. Reading all points...')
                pcd = las.points
            else:
                try:
                    if hasattr(las, 'classification'):
                        print(f'- Reading points with classification value {classification}...')
                        pcd = las.points[las.raw_classification == classification]
                    else:
                        raise ValueError(f'The input file does not contain classification values.')
                except ValueError as error:
                    message = str(error)
                    lines = message.split('\n')
                    print(lines[-2])
                    print(lines[-1])
                    exit()

            if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
                print(f'- Reading RGB values...')
                points = np.vstack((pcd.x, pcd.y, pcd.z, pcd.red / 255.0, pcd.green / 255.0, pcd.blue / 255.0)).transpose()
            else:
                print(f'- RGB values are not provided. Reading only XYZ values...')
                points = np.vstack((pcd.x, pcd.y, pcd.z)).transpose()
            

        end = time.time()
        print(f'File reading is completed in {end - start:.2f} seconds. The point cloud contains {points.shape[0]} points.\n')
        return points, pdal_points

    def applyFilters(self, pdal_points: np.ndarray, filters=[], multi_array=False) -> np.ndarray:
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

    def smrf(self, pdal_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        start = time.time()
        print(f'Applying SMRF algorithm...')
        smrf = pdal.Filter.smrf()
        reclass = pdal.Filter.assign(value='Classification = 0')
        filter_list = [reclass, smrf]
        pdal_points = SamLidar.applyFilters(self, pdal_points, filter_list)
        #print('reclassed',points)
        #points = SamLidar.applyFilters(self, points, [smrf])
        #print('smrfed',points)
        #pipeline = pdal.Writer.las(filename="ground_test.las").pipeline(points)
        #pipeline.execute()
        ground = np.where(pdal_points["Classification"]==2)
        non_ground = np.where(pdal_points["Classification"]!=2)
        #pdal_points = np.delete(pdal_points, ground)
        x = pdal_points["X"]
        y = pdal_points["Y"]
        z = pdal_points["Z"]
        points = np.vstack((x, y, z)).T
        points = np.delete(points, ground, 0)
        #pipe = pdal.Writer.las(filename="smrf_test.las").pipeline(pdal_points)
        #pipe.execute()
        #dtypes = pdal_points.dtype
        #fields = tuple(dtypes.names)

        ground = np.array(ground)
        non_ground = np.array(non_ground)
        ground = np.ravel(ground)
        non_ground = np.ravel(non_ground)
        end = time.time()
        print(f'SMRF algorithm is completed in {end - start:.2f} seconds. The filtered non-ground cloud contains {points.shape[0]} points.\n')
        return points, non_ground, ground, pdal_points #structured_to_unstructured(pdal_points)

    def segment(self, points: np.ndarray, text_prompt: str = None, image_path: str = 'raster.tif', labels_path: str = 'labeled.tif') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Segments a point cloud based on the provided parameters and returns the segment IDs, original image, and segmented image.

        :param points: The point cloud data as a NumPy array.
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
                sam.predict(image=image_path, text_prompt=text_prompt, box_threshold=0.24, text_threshold=0.3, output=labels_path)
                print(f'- Saving segmented image...')
            else:
                sam = self.sam_geo
                sam.generate(source=image_path, output=labels_path, erosion_kernel=(3, 3), foreground=True, unique=True)
                print(f'- Saving segmented image...')

        with rasterio.open(labels_path, 'r') as src:
            segmented_image = src.read()
            segmented_image = np.squeeze(segmented_image)

        print(f'- Generating segment IDs...')
        segment_ids = image_to_cloud(points, minx, maxy, segmented_image, self.resolution)
        end = time.time()

        print(f'Segmentation is completed in {end - start:.2f} seconds. Number of instances: {np.max(segmented_image)}\n')
        return segment_ids, segmented_image, image_rgb
  
    
    def grouping(self, pdal_points: np.ndarray, segment_ids: np.ndarray, ground, non_ground) -> np.ndarray:
        #add support for ground = none
        if ground is not None:
            segment_ids = np.append(segment_ids, np.full(len(ground), -1))
        
        indices = np.append(non_ground, ground)
        points_ordered = np.take(pdal_points, indices)
        temp = np.full(points_ordered.shape[0], 1, dtype=[('seg_id', 'i4')])
        merged = rfn.merge_arrays((points_ordered, temp), flatten=True)
        
        merged['seg_id']=segment_ids
        #pdal_points = np.sort(pdal_points, order="seg_id")

        #unique=np.unique(pdal_points['seg_id'])
        #grouped_arrays = np.split

        groupby = pdal.Filter.groupby(dimension="seg_id")
        filters = [groupby]
        points_grouped = SamLidar.applyFilters(self, merged, filters, multi_array=True)

        return points_grouped
    
    def featureFilter(self, points_grouped: np.ndarray) -> np.ndarray:
        breakpoint()
        cov = pdal.Filter.covariancefeatures(feature_set="Scattering,Planarity,Verticality")
        eigen = pdal.Filter.
        filters = [cov]
        points_filtered = []
        for arr in points_grouped:
            filtered = SamLidar.applyFilters(self, arr, filters)
            points_filtered = points_filtered.append(filtered)
            breakpoint()
        breakpoint()
        return filtered


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