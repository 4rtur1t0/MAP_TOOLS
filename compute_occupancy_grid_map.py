"""
    Builds an occupancy grid map by detecting obstacles with a difference of normals approach.
    Caution: the whole map is processed on a single batch. The map is previously filtered on Z.

    Author: Arturo Gil.
    Date: 07/2025

"""
import open3d as o3d
import numpy as np
from PIL import Image


def select_local_pointcloud(pointcloud_global, lx, ly, lz):
    pointcloud_global_points = np.asarray(pointcloud_global.points)

    maskx = (pointcloud_global_points[:, 0] > lx[0]) & (pointcloud_global_points[:, 0] < lx[1])
    masky = (pointcloud_global_points[:, 1] > ly[0]) & (pointcloud_global_points[:, 1] < ly[1])
    maskz = (pointcloud_global_points[:, 2] > lz[0]) & (pointcloud_global_points[:, 2] < lz[1])
    mask = maskx & masky & maskz
    indices_in_global_map = np.where(mask)[0]
    pointcloud_local = pointcloud_global.select_by_index(list(indices_in_global_map), invert=False)
    return pointcloud_local, indices_in_global_map


def difference_of_normals(pointcloud, small_radius, large_radius):
    pointcloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=small_radius, max_nn=50))
    # Copy to preserve the normals for small radius
    normals_small = np.asarray(pointcloud.normals).copy()
    pointcloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=large_radius, max_nn=50))
    normals_large = np.asarray(pointcloud.normals).copy()
    # Compute the Difference of Normals (DoN)
    diff_norm = (normals_large - normals_small) / 2
    return diff_norm, normals_small, normals_large


def filter_edge_indices(don_abs, data, threshold, max_distance, min_distance):
    distances = np.linalg.norm(data, axis=1)
    mask = (don_abs > threshold) & (distances < max_distance) & (distances > min_distance)
    edge_indices = np.where(mask)[0]
    return edge_indices


def filter_collision_indices(min_distance, margin, data):
    if min_distance > 2:
        return np.array([], dtype=int)
    size_x, size_y, size_z = 0.7, 1, 1.25
    wheel_radius = 0.35463
    x, y, z = np.abs(data[:, 0]), np.abs(data[:, 1]), data[:, 2]
    lidar_height = 1.1325
    mask = ((x - margin < size_x / 2) & (y - margin < size_y / 2) &
            (wheel_radius < (lidar_height + z)) & ((lidar_height + z) < size_z))
    collision_indices = np.where(mask)[0]
    return collision_indices


def find_obstacles(pointcloud):
    """
    Compute obstacles with a difference of gaussians computed at two different radii
    """
    small_radius = 0.2
    large_radius = 0.5
    threshold_don = 0.05

    print('Computing difference of Normals')
    # compute a difference of normals
    diff_norm, normals_small, normals_large = difference_of_normals(pointcloud, small_radius, large_radius)
    don_abs = np.linalg.norm(diff_norm, axis=1)
    mask = (don_abs > threshold_don)

    all_idx = np.arange(len(pointcloud.points))
    idx_obstacles = all_idx[mask]
    pcd_obstacles = pointcloud.select_by_index(idx_obstacles)
    pcd_obstacles.paint_uniform_color([0.0, 0.0, 0.0])
    o3d.visualization.draw_geometries([pointcloud, pcd_obstacles])
    return pcd_obstacles


def save_to_image(global_pcd, filename='occupancy_map.png'):
    """
    Save the pcd to a 2D image.
    """
    # Parámetros del mapa de ocupación
    resolution = 0.05  # metros por celda
    padding = 50  # celdas extra alrededor

    points = np.asarray(global_pcd.points)

    # 2. Proyectar a 2D (plano XY), opcionalmente filtrar por altura Z
    # z_min, z_max = -0.5, 1.0  # Altura para filtrar el suelo/techo
    # filtered_points = points[(points[:, 2] > z_min) & (points[:, 2] < z_max)]
    xy_points = points[:, :2]

    # Obtener límites
    min_xy = xy_points.min(axis=0) - resolution * padding
    max_xy = xy_points.max(axis=0) + resolution * padding

    # Tamaño del mapa
    map_size = np.ceil((max_xy - min_xy) / resolution).astype(int)
    # create white map
    map_grid = 255*np.ones(map_size[::-1], dtype=np.uint8)  # [rows, cols]

    # 4. Convertir coordenadas a índices de celda
    indices = np.floor((xy_points - min_xy) / resolution).astype(int)
    indices = np.clip(indices, 0, np.array(map_size) - 1)

    # 5. Marcar ocupación
    for i in indices:
        y_idx, x_idx = map_size[1] - i[1] - 1, i[0]  # invertir eje Y para visualización
        # cross, also add
        map_grid[y_idx, x_idx] = 0  # ocupado (negro)
        map_grid[y_idx+1, x_idx] = 0  # ocupado (negro)
        map_grid[y_idx, x_idx+1] = 0  # ocupado (negro)
        map_grid[y_idx+1, x_idx+1] = 0  # ocupado (negro)
        map_grid[y_idx - 1, x_idx] = 0  # ocupado (negro)
        map_grid[y_idx, x_idx - 1] = 0  # ocupado (negro)
        map_grid[y_idx - 1, x_idx - 1] = 0  # ocupado (negro)

    # 6. Guardar como imagen PNG
    img = Image.fromarray(map_grid, mode='L')
    img.save(filename)


def build_occupancy_grid_map():
    # Read the final transform (i.e. via GraphSLAM)
    # You may be using different estimations to build the map: i.e. scanmatching or the results from graphSLAM
    # select as desired
    # directory = '/media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/IO2-2025-03-25-16-54-17'
    # directory = '/media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/IO3-2025-06-16-13-49-28'
    # directory = '/media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/IO4-2025-06-16-15-56-11'
    map_directory = '/media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/IO5-2025-06-16-17-53-54'
    ###########
    input_map_filename = 'global_map_filtered.pcd'
    output_occupancy_map_filename = 'occupancy_map.png'

    # 1. Load the global map (PCD)
    map_filename = map_directory + '/' + input_map_filename
    print('LOADING MAP: ', map_filename)
    pointcloud_global = o3d.io.read_point_cloud(map_filename)
    print("[INFO] Global map loaded with", len(pointcloud_global.points), "points.")

    # Remove high points. Get a submap
    p = [0, 0, 0]
    submap_edge = 500
    submap_height = 1.0
    lx = [p[0] - submap_edge, p[0] + submap_edge]
    ly = [p[1] - submap_edge, p[1] + submap_edge]
    lz = [p[2] - submap_height, p[2] + submap_height]
    pointcloud_global, indices_pcd = select_local_pointcloud(pointcloud_global=pointcloud_global, lx=lx, ly=ly, lz=lz)
    o3d.visualization.draw_geometries([pointcloud_global])

    # Now, find obstacles
    pointcloud_global_obstacles = find_obstacles(pointcloud_global)
    # save to occupancy grid map
    save_to_image(global_pcd=pointcloud_global_obstacles, filename=output_occupancy_map_filename)


if __name__ == '__main__':
    build_occupancy_grid_map()

