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
    directory = '/media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/IO5-2025-06-16-17-53-54'
    ###########
    output_map_filename = 'global_map.pcd'
    # use, for example, 1 out of 5 LiDARS to build the map
    keyframe_sampling = 20
    # keyframe_sampling = 500
    # use, for example, voxel_size=0.2. Use voxel_size=None to use full resolution
    # we are voxelizing the local scans and the resulting global map as well
    voxel_size = 0.05
    maplidar = Map()
    # read the data in TUM format
    maplidar.read_data_tum(directory=directory, filename='/robot0/SLAM/data_poses_tum.txt')

    # draw the map now. # or build and write pcd!
    # maplidar.draw_map(keyframe_sampling=keyframe_sampling, voxel_size=voxel_size)


    # caution, voxelizing the local clouds, but not the global pointcloud
    # voxel size is a needed discretization of the local lidar
    radii = [0.5, 50]
    heights = [-2, 100]
    pointcloud_global = maplidar.build_map(keyframe_sampling=keyframe_sampling,
                                           voxel_size=voxel_size, radii=radii,
                                           heights=heights)
    # remove, then downsample...
    print('GLOBAL POINTCLOUD INFO:')
    print(pointcloud_global)
    # also downsample the global map obtained
    # pointcloud_global_sampled = pointcloud_global.voxel_down_sample(voxel_size=voxel_size)
    # print('GLOBAL POINTCLOUD INFO after voxelization:')
    # print(pointcloud_global_sampled)

    # paint
    o3d.visualization.draw_geometries([pointcloud_global])

    # save the result to a pcd file
    print('Saving sampled global map to file: ', output_map_filename)
    maplidar.save_pcd_to_file(pointcloud=pointcloud_global, filename=directory + '/' + output_map_filename)


if __name__ == '__main__':
    map_viewer()

