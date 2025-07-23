"""
    Filter the map.
    The map is filtered near the path followed by the robot when constructing the map.
    Low density areas are removed. Typically, movin objects, such as people wandering around

    Author: Arturo Gil.
    Date: 07/2025

    TTD: Save map for future use in MCL localization.
         The map can be now saved in PCL format in order to use o3D directly.
         Also, the map can be saved in a series of pointclouds along with their positions, however path planning using,
         for example, PRM, may not be direct
"""
from map.map import Map
import open3d as o3d
import numpy as np
from observations.posesarray import PosesArray
# import matplotlib.pyplot as plt


class DistChecker():
    def __init__(self, p0, d_th=5.0):
        self.p0 = p0
        self.d_th = d_th

    def check_dist(self, pi):
        d = np.linalg.norm(self.p0-pi)
        if d > self.d_th:
            self.p0 = pi
            return True
        return False


def select_local_pointcloud(pointcloud_global, lx, ly, lz):
    pointcloud_global_points = np.asarray(pointcloud_global.points)

    maskx = (pointcloud_global_points[:, 0] > lx[0]) & (pointcloud_global_points[:, 0] < lx[1])
    masky = (pointcloud_global_points[:, 1] > ly[0]) & (pointcloud_global_points[:, 1] < ly[1])
    maskz = (pointcloud_global_points[:, 2] > lz[0]) & (pointcloud_global_points[:, 2] < lz[1])
    mask = maskx & masky & maskz
    indices_in_global_map = np.where(mask)[0]
    pointcloud_local = pointcloud_global.select_by_index(list(indices_in_global_map), invert=False)
    return pointcloud_local, indices_in_global_map


def map_filtering():
    # Read the final transform (i.e. via GraphSLAM)
    # You may be using different estimations to build the map: i.e. scanmatching or the results from graphSLAM
    # select as desired
    # directory = '/media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/IO2-2025-03-25-16-54-17'
    # directory = '/media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/IO3-2025-06-16-13-49-28'
    # directory = '/media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/IO4-2025-06-16-15-56-11'
    map_directory = '/media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/IO5-2025-06-16-17-53-54'
    ###########
    input_map_filename = 'global_map.pcd'
    output_map_filename = 'global_map_filtered.pcd'

    # skip transform
    # skip_transforms = 50
    local_submap_edge = 5.0
    show = False
    # DYNAMIC OBJECT FILTERING:  at least nb_points in a certain radius
    nb_points = 20
    radius = 0.1
    # DOWNSAMLING: AFTER THAT, downsample with this voxel size
    # After that, downsample with this voxel size and save the final map
    voxel_size = 0.1

    # 1. Load the global map (PCD)
    map_filename = map_directory + '/' + input_map_filename
    print('LOADING MAP: ', map_filename)
    pointcloud_global = o3d.io.read_point_cloud(map_filename)
    print("[INFO] Global map loaded with", len(pointcloud_global.points), "points.")
    o3d.visualization.draw_geometries([pointcloud_global])

    # ALGORITHM: we will get local submaps (square) around the known path that builds the map.
    # At each local submap. Filter the points with low point density. Store the indices, remove the points from tha map
    # save the map.
    robotpath = PosesArray()
    robotpath.read_data_tum(directory=map_directory, filename='/robot0/SLAM/data_poses_tum.txt')
    global_transforms = robotpath.get_transforms()
    all_indices_outliers = []
    # the distchecker is used to measure distance and get the submaps when needed
    dist_checker = DistChecker(global_transforms[0].pos(), d_th=local_submap_edge)
    for i in range(len(global_transforms)):
        # print('Processing submap i: ', i, 'out of', len(global_transforms))
        transform = global_transforms[i]
        pi = transform.pos()
        # process the first submap, then check distance
        if not dist_checker.check_dist(pi) and (i > 0):
            continue
        print('Processing submap i: ', i, 'out of', len(global_transforms))
        # OBTAIN A SUBMAP OF THE GLOBAL MAP
        lx = [pi[0]-local_submap_edge, pi[0]+local_submap_edge]
        ly = [pi[1]-local_submap_edge, pi[1]+local_submap_edge]
        lz = [pi[2]-local_submap_edge, pi[2]+local_submap_edge]
        pointcloud_local, idx_pointcloud_local = select_local_pointcloud(pointcloud_global=pointcloud_global,
                                                                         lx=lx, ly=ly,
                                                                         lz=lz)
        print('Now filtering')
        # Radius outlier removal. Remove the points WHICH DO NOT HAVE nb_points in a radius near it
        cl_inlier, indices_local_inliers = pointcloud_local.remove_radius_outlier(nb_points=nb_points, radius=radius)
        indices_local_inliers = np.array(indices_local_inliers)
        all_indices = np.arange(len(pointcloud_local.points))
        # obtain the outliers by difference
        indices_local_outliers = np.setdiff1d(all_indices, indices_local_inliers)
        # propagate to global indices
        indices_global_outliers = idx_pointcloud_local[indices_local_outliers]
        # save the list of outliers
        all_indices_outliers.extend(indices_global_outliers)
        # local visualization
        if show:
            inlier_cloud_temp = pointcloud_local.select_by_index(indices_local_inliers)
            o3d.visualization.draw_geometries([inlier_cloud_temp])
            outlier_cloud_temp = pointcloud_local.select_by_index(indices_local_outliers)
            o3d.visualization.draw_geometries([outlier_cloud_temp])

    outlier_cloud = pointcloud_global.select_by_index(all_indices_outliers, invert=False)
    outlier_cloud.paint_uniform_color([0.5, 0.5, 0.5])
    # o3d.visualization.draw_geometries([outlier_cloud])
    # Generate the pointcloud by inverting the outliers. This pointcloud will not include the outliers
    inlier_cloud = pointcloud_global.select_by_index(all_indices_outliers, invert=True)
    # o3d.visualization.draw_geometries([inlier_cloud])
    # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

    print('Number of points in the original pointcloud')
    print(pointcloud_global)
    print('Number of points in the filtered inlier pointcloud')
    print(inlier_cloud)
    print('SAMPLING THE GLOBAL POINTCLOUD. This may take a while...')
    pointcloud_global_sampled = inlier_cloud.voxel_down_sample(voxel_size=voxel_size)
    print('GLOBAL POINTCLOUD INFO after voxelization:')
    print(pointcloud_global_sampled)

    # view the sampled pointcloud, place the outliers as well
    o3d.visualization.draw_geometries([pointcloud_global_sampled, outlier_cloud])
    # only the filtered pointcloud now
    o3d.visualization.draw_geometries([pointcloud_global_sampled])

    # save the result to a pcd file
    print('Saving sampled global map to file: ', output_map_filename)
    o3d.io.write_point_cloud(filename=map_directory + '/' + output_map_filename, pointcloud=pointcloud_global_sampled,
                             print_progress=True)


if __name__ == '__main__':
    map_filtering()

