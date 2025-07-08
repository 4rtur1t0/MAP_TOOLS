import numpy as np
import yaml
from lidarscanarray.lidarscanarray import LiDARScanArray
from observations.posesarray import PosesArray, ArucoPosesArray, ArucoLandmarksPosesArray
import tilemapbase
import matplotlib.pyplot as plt
from tools.gpsconversions import utm2gps
import pyproj
import pandas as pd
from tools.plottools import plot_gps_OSM
from PIL import Image
import open3d as o3d

from config import PARAMETERS


class Map():
    """
    A Map!
    The map is composed by:
    - A list of estimated poses X as the robot followed a path.
    - A LiDAR scan associated to each pose.
    - A number of ARUCO landmarks. This part is optional, however



    A number of methods are included in this class:
    - Plotting the global pointcloud, including the path and landmarks.
    - Plotting the pointcloud

    - Includes a method to localize on the map


    """
    def __init__(self):
        """
        given a list of scan times (ROS times), each pcd is read on demand
        """
        self.robotpath = None
        self.lidarscanarray = None
        self.landmarks = None
        self.utm_ref = None

    def __len__(self):
        return len(self.robotpath)

    def read_data(self, directory):
        """
        Read the estimated path of the robot, landmarks and LiDAR scans.
        the map is formed by a set of poses, each pose associated to a pointcloud.
        no global map is built and stored. Though, the view_map method allows
        """
        self.robotpath = PosesArray()
        self.robotpath.read_data(directory=directory, filename='/robot0/SLAM/solution_graphslam_lidar.csv')
        # Load the LiDAR scan array. Each pointcloud with its associated time.
        # Each lidar scan is associated to a given pose in the robotpath
        self.lidarscanarray = LiDARScanArray(directory=directory)
        # self.lidarscanarray.read_parameters()
        self.lidarscanarray.parameters = PARAMETERS.config
        self.lidarscanarray.read_data()
        # remove scans without corresponding odometry (in consequence, without scanmatching)
        self.lidarscanarray.remove_orphan_lidars(pose_array=self.robotpath)
        # lidarscanarray.remove_orphan_lidars(pose_array=smobsarray)
        # load the scans according to the times, do not load the corresponding pointclouds
        self.lidarscanarray.add_lidar_scans()
        # also load the ARUCO landmarks
        # self.landmarks_aruco = ArucoLandmarksPosesArray()
        # self.landmarks_aruco.read_data(directory=directory, filename='/robot0/SLAM/solution_graphslam_landmarks.csv')

        # global_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_file_global = directory + '/' + 'robot0/gps0/reference.yaml'
        with open(yaml_file_global) as file:
            self.utm_ref = yaml.load(file, Loader=yaml.FullLoader)

    def read_data_tum(self, directory, filename='/robot0/SLAM/data_poses_tum.txt'):
        """
        Read the estimated path of the robot in tum format
        """
        self.robotpath = PosesArray()
        self.robotpath.read_data_tum(directory=directory, filename=filename)
        # Load the LiDAR scan array. Each pointcloud with its associated time.
        # Each lidar scan is associated to a given pose in the robotpath
        self.lidarscanarray = LiDARScanArray(directory=directory)
        # self.lidarscanarray.read_parameters()
        self.lidarscanarray.parameters = PARAMETERS.config.get('scanmatcher')
        self.lidarscanarray.read_data()
        self.lidarscanarray.add_lidar_scans()

    def draw_all_clouds(self):
        # self.lidarscanarray.draw_all_clouds_visualizer()
        self.lidarscanarray.draw_all_clouds()

    def draw_map(self, voxel_size, keyframe_sampling=20):
        """
        Possibilities:
        - view path
        - view pointclouds
        - view ARUCO landmarks
        - The terraplanist option sets artificially z=0 in the solution.
        """
        global_transforms = self.robotpath.get_transforms()
        self.lidarscanarray.draw_map(global_transforms=global_transforms,
                                     voxel_size=voxel_size,
                                     radii=[0.5, 120],
                                     heights=[-2, 50],
                                     keyframe_sampling=keyframe_sampling)

    def build_map(self, voxel_size, keyframe_sampling=20, filename='global_map.pcd'):
        """
        Builds a global pcd map and writes to disk.
        """
        global_transforms = self.robotpath.get_transforms()
        global_pcd = self.lidarscanarray.build_map(global_transforms=global_transforms,
                                                   voxel_size=voxel_size,
                                                   radii=[0.5, 120],
                                                   heights=[-2, 50],
                                                   keyframe_sampling=keyframe_sampling)

        # if delete_dynamic_objects:
        #     global_pcd = self.lidarscanarray.delete_empty_spaces_in_map(pointcloud_global=global_pcd,
        #                                                global_transforms=global_transforms,
        #                                                voxel_size=voxel_size,
        #                                                radii=[0.5, 120],
        #                                                heights=[-2, 50],
        #                                                keyframe_sampling=keyframe_sampling)
        if filename:
            o3d.io.write_point_cloud(filename=filename, pointcloud=global_pcd, print_progress=True)
        return global_pcd

    def delete_empty_spaces_in_map(self, pointcloud_global, voxel_size, radii,
                                   heights,  keyframe_sampling):
        global_transforms = self.robotpath.get_transforms()
        global_pcd = self.lidarscanarray.delete_empty_spaces_in_map(pointcloud_global=pointcloud_global,
                                                       global_transforms=global_transforms,
                                                       voxel_size=voxel_size,
                                                       radii=radii,
                                                       heights=heights,
                                                       keyframe_sampling=keyframe_sampling)
        return global_pcd

    def build_occupancy_grid_map(self, filename, voxel_size, keyframe_sampling=20):
        """
        Possibilities:
        - view path
        - view pointclouds
        - view ARUCO landmarks
        - The terraplanist option sets artificially z=0 in the solution.
        """
        global_transforms = self.robotpath.get_transforms()
        # caution. Build the map by limiting radii to a local context
        global_pcd = self.lidarscanarray.build_map(global_transforms=global_transforms,
                                                   voxel_size=voxel_size,
                                                   radii=[0.5, 15],
                                                   heights=[-0.7, 1.0],
                                                   keyframe_sampling=keyframe_sampling)
        self.save_to_image(global_pcd, filename=filename)

    def save_to_image(self, global_pcd, filename='occupancy_map.png'):
        # 3. Parámetros del mapa
        resolution = 0.05  # metros por celda
        padding = 50  # celdas extras alrededor

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


    def plot_solution_utm_OSM(self, expand=0.001, save_fig=False):
        # the global transforms expressed
        global_transforms = self.robotpath.get_transforms()
        utm_x = []
        utm_y = []
        utm_altitude = []
        for transform in global_transforms:
            pos = transform.pos()
            utm_x.append(pos[0])
            utm_y.append(pos[1])
            utm_altitude.append(pos[2])

        # convert from UTM to GPS to plot
        latitude, longitude = utm2gps(utm_x, utm_y)
        latitude = np.array(latitude)
        longitude = np.array(longitude)
        # add ref
        latitude = latitude + self.utm_ref.get('latitude')
        longitude = longitude + self.utm_ref.get('longitude')
        # now project lat, lng to osm map.
        tilemapbase.init(create=True)
        extent = tilemapbase.Extent.from_lonlat(
            longitude.min() - expand,
            longitude.max() + expand,
            latitude.min() - expand,
            latitude.max() + expand,
        )
        x = []
        y = []
        for i in range(len(latitude)):
            xi, yi = tilemapbase.project(longitude[i], latitude[i])
            x.append(xi)
            y.append(yi)
        # trip_projected = df_gps.apply(
        #     lambda x: tilemapbase.project(longitude, x.latitude), axis=1
        # ).apply(pd.Series)
        # trip_projected.columns = ["x", "y"]

        tiles = tilemapbase.tiles.build_OSM()
        fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        plotter = tilemapbase.Plotter(extent, tiles, height=600)
        plotter.plot(ax, tiles, alpha=0.8)
        # ax.plot(trip_projected.x, trip_projected.y, color='blue', linewidth=1)
        ax.scatter(x, y, color='blue', marker='.', s=1)
        plt.axis('off')
        plt.show(block=True)
        if save_fig:
            fig.savefig('trip.png', bbox_inches='tight', pad_inches=0, dpi=300)


    def plot_test(self):
        # Reference point (arbitrary origin in latitude, longitude)
        ref_lat = self.utm_ref.get('latitude')  # Latitude of the reference point
        ref_lon = self.utm_ref.get('longitude')  # Longitude of the reference point

        # UTM coordinates relative to the reference point (arbitrary origin)
        # utm_x = np.array([500000, 500100, 500200])  # Easting (X)
        # utm_y = np.array([4649776, 4649876, 4649976])  # Northing (Y)

        global_transforms = self.robotpath.get_transforms()
        utm_x = []
        utm_y = []
        utm_altitude = []
        for transform in global_transforms:
            pos = transform.pos()
            utm_x.append(pos[0])
            utm_y.append(pos[1])
            utm_altitude.append(pos[2])
        utm_x = np.array(utm_x)
        utm_y = np.array(utm_y)

        # (proj='utm', zone='30', ellps='WGS84', datum='WGS84', preserve_units=False,
        # units='m')

        # Define the UTM zone and projection (use 'epsg:32633' for UTM Zone 33N)
        utm_zone = 30
        utm_proj = pyproj.CRS(f"EPSG:326{utm_zone}")

        # Define the WGS84 (GPS) projection
        wgs84_proj = pyproj.CRS("EPSG:4326")

        # Initialize a transformer object to convert from UTM to WGS84 (GPS)
        transformer = pyproj.Transformer.from_crs(utm_proj, wgs84_proj, always_xy=True)

        # First, we need to transform the reference point to UTM coordinates
        geod = pyproj.Geod(ellps="WGS84")
        ref_utm_x, ref_utm_y, _ = geod.fwd(ref_lon, ref_lat, 0, 0)  # Use forward projection to get UTM

        # Adjust UTM coordinates by subtracting the reference UTM coordinates
        utm_x_adjusted = utm_x - ref_utm_x
        utm_y_adjusted = utm_y - ref_utm_y

        # Now, transform the adjusted UTM coordinates to latitude and longitude
        longitude, latitude = transformer.transform(utm_x_adjusted, utm_y_adjusted)

        df_gps = pd.DataFrame({'longitude': longitude, 'latitude': latitude})
        plot_gps_OSM(df_gps, expand=0.001, save_fig=False)

        # # Create a plot with a tilemap base
        # fig, ax = plt.subplots(figsize=(8, 8))
        #
        # # Create an OpenStreetMap basemap
        # tilemapbase.tiles.plot(ax=ax, zoom=12, lon_0=ref_lon, lat_0=ref_lat, zorder=1)
        #
        # # Plot the transformed UTM coordinates (latitude, longitude)
        # ax.scatter(longitude, latitude, color='red', zorder=2, label="Points")
        #
        # # Add a label for each point (optional)
        # for lon, lat in zip(longitude, latitude):
        #     ax.text(lon + 0.001, lat + 0.001, f"({lon:.2f}, {lat:.2f})", fontsize=8)
        #
        # # Set labels and title
        # ax.set_xlabel('Longitude')
        # ax.set_ylabel('Latitude')
        # ax.set_title('UTM Coordinates on OpenStreetMap')
        #
        # # Show the map
        # plt.legend()
        # plt.show()