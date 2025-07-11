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


def map_viewer():
    # Read the final transform (i.e. via GraphSLAM)
    # You may be using different estimations to build the map: i.e. scanmatching or the results from graphSLAM
    # select as desired
    # directory = '/media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/IO2-2025-03-25-16-54-17'
    # directory = '/media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/IO3-2025-06-16-13-49-28'
    # directory = '/media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/IO4-2025-06-16-15-56-11'
    directory = '/media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/IO5-2025-06-16-17-53-54'
    ###########
    output_map_filename = 'global_map.pcd'
    # use, for example, 1 out of 5 LiDARS to build the map
    keyframe_sampling = 20
    # keyframe_sampling = 500
    # use, for example, voxel_size=0.2. Use voxel_size=None to use full resolution
    voxel_size = 0.2
    maplidar = Map()
    # read the data in TUM format
    maplidar.read_data_tum(directory=directory, filename='/robot0/SLAM/data_poses_tum.txt')

    # visualize the map on the UTM reference frame
    # maplidar.draw_map(keyframe_sampling=keyframe_sampling, voxel_size=voxel_size)
    # or build and write pcd!
    pointcloud_global = maplidar.build_map(keyframe_sampling=keyframe_sampling,
                                           voxel_size=voxel_size)
    maplidar.save_pcd_to_file(pointcloud=pointcloud_global, filename=directory + '/' + output_map_filename)
    # now remove dynamic objects.
    # if known, dynamic objects in this case are people, which should be at known heights
    # radii = [0.5, 30]
    # heights = [-2, 2]
    # keyframe_sampling = 100
    # global_pcd = maplidar.delete_empty_spaces_in_map(pointcloud_global=pointcloud_global,
    #                                                  voxel_size=voxel_size,
    #                                                             radii=radii,
    #                                                             heights=heights,
    #                                                             keyframe_sampling=keyframe_sampling)

    # MODIFY!

    # or build and write occupancy grid map!
    # maplidar.build_occupancy_grid_map(filename=directory + '/occupancy_map.png',
    #                                   keyframe_sampling=keyframe_sampling,
    #                                   voxel_size=voxel_size)


if __name__ == '__main__':
    map_viewer()

