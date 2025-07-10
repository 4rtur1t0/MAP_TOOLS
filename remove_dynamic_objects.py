"""
    Visualize map from known/ground truth trajectory and LIDAR.

    Author: Arturo Gil.
    Date: 03/2024

    TTD: Save map for future use in MCL localization.
         The map can be now saved in PCL format in order to use o3D directly.
         Also, the map can be saved in a series of pointclouds along with their positions, however path planning using,
         for example, PRM, may not be direct
"""
from map.map import Map
import open3d as o3d

def map_viewer():
    # Read the final transform (i.e. via GraphSLAM)
    # You may be using different estimations to build the map: i.e. scanmatching or the results from graphSLAM
    # select as desired
    # directory = '/media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/IO2-2025-03-25-16-54-17'
    # directory = '/media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/IO3-2025-06-16-13-49-28'
    # directory = '/media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/IO4-2025-06-16-15-56-11'
    map_directory = '/media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/IO5-2025-06-16-17-53-54'
    ###########
    input_map_filename = 'global_map.pcd'
    output_map_filename = 'global_map_clean.pcd'
    # use, for example, 1 out of 5 LiDARS to build the map
    # keyframe_sampling = 20
    # keyframe_sampling = 500
    # use, for example, voxel_size=0.2. Use voxel_size=None to use full resolution
    # voxel_size = 0.2
    maplidar = Map()
    # read the data in TUM format
    maplidar.read_data_tum(directory=map_directory, filename='/robot0/SLAM/data_poses_tum.txt')

    # visualize the map on the UTM reference frame
    # maplidar.draw_map(keyframe_sampling=keyframe_sampling, voxel_size=voxel_size)
    # or build and write pcd!
    # pointcloud_global = maplidar.build_map(keyframe_sampling=keyframe_sampling,
    #                                        voxel_size=voxel_size, filename=directory + '/' + map_filename)

    # 1. Load the global map (PCD)
    map_filename = map_directory + '/' + input_map_filename
    print('LOADING MAP: ', map_filename)
    pointcloud_global = o3d.io.read_point_cloud(map_filename)
    print("[INFO] Global map loaded with", len(pointcloud_global.points), "points.")
    # 3. Downsample for faster processing the global map
    voxel_size_global_map = 0.2  # adjust as needed
    pointcloud_global.voxel_down_sample(voxel_size_global_map)

    # now remove dynamic objects.
    # if known, dynamic objects in this case are people, which should be at known heights
    voxel_size = 0.2
    radii = [0.5, 2]
    heights = [-0.7, 1.5]
    keyframe_sampling = 10
    global_pcd_clean = maplidar.delete_empty_spaces_in_map(pointcloud_global=pointcloud_global,
                                                     voxel_size=voxel_size,
                                                                radii=radii,
                                                                heights=heights,
                                                                keyframe_sampling=keyframe_sampling)

    maplidar.save_pcd_to_file(pointcloud=global_pcd_clean, filename=map_directory + '/' + output_map_filename)
    # MODIFY!

    # or build and write occupancy grid map!
    # maplidar.build_occupancy_grid_map(filename=directory + '/occupancy_map.png',
    #                                   keyframe_sampling=keyframe_sampling,
    #                                   voxel_size=voxel_size)


if __name__ == '__main__':
    map_viewer()

