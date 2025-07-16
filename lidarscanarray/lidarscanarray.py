import time
import bisect
import numpy as np
from artelib.homogeneousmatrix import HomogeneousMatrix
import open3d as o3d

from artelib.rotationmatrix import RotationMatrix
from lidarscanarray.lidarscan import LiDARScan
from eurocreader.eurocreader import EurocReader
from tools.sampling import sample_times
import yaml
import matplotlib.pyplot as plt


class LiDARScanArray:
    def __init__(self, directory):
        """
        given a list of scan times (ROS times), each pcd is read on demand
        """
        self.directory = directory
        self.scan_times = None
        self.lidar_scans = []
        self.show_registration_result = False
        self.parameters = None

    def __len__(self):
        return len(self.scan_times)

    def __getitem__(self, item):
        return self.lidar_scans[item]

    def load_parameters(self, parameters):
        self.parameters = parameters

    def read_parameters(self):
        yaml_file_global = self.directory + '/' + 'robot0/scanmatcher_parameters.yaml'
        with open(yaml_file_global) as file:
            scanmatcher_parameters = yaml.load(file, Loader=yaml.FullLoader)
        self.parameters = scanmatcher_parameters
        return scanmatcher_parameters

    def read_data(self):
        euroc_read = EurocReader(directory=self.directory)
        df_lidar = euroc_read.read_csv(filename='/robot0/lidar/data.csv')
        scan_times = df_lidar['#timestamp [ns]'].to_numpy()
        self.scan_times = scan_times

    def sample_data(self):
        start_index = self.parameters.get('start_index', 0)
        delta_time = self.parameters.get('delta_time', None)
        scan_times = sample_times(sensor_times=self.scan_times, start_index=start_index, delta_time=delta_time)
        self.scan_times = scan_times

    def get_time(self, index):
        """
        Get the time for a corresponding index
        """
        return self.scan_times[index]

    def get_index_closest_to_time(self, timestamp, delta_threshold_s):
        """
        Given a timestamp. Find the closest time within delta_threshold_s seconds and return its corrsponding index.
        """
        idx1, t1, idx2, t2 = self.find_closest_times_around_t_bisect(timestamp)
        d1 = abs((timestamp - t1) / 1e9)
        d2 = abs((t2 - timestamp) / 1e9)
        print('Time ARUCO times: ', d1, d2)
        if (d1 > delta_threshold_s) and (d2 > delta_threshold_s):
            print('get_index_closest_to_time could not find any close time')
            return None, None
        if d1 <= d2:
            return idx1, t1
        else:
            return idx2, t2
        # return None

    def get_times(self):
        return self.scan_times

    def get(self, index):
        return self.lidar_scans[index]

    def find_closest_times_around_t_bisect(self, t):
        # Find the index where t would be inserted in sorted_times
        idx = bisect.bisect_left(self.scan_times, t)

        # Determine the two closest times
        if idx == 0:
            # t is before the first element
            return 0, self.scan_times[0], 1, self.scan_times[1]
        elif idx == len(self.scan_times):
            # t is after the last element
            return -2, self.scan_times[-2], -1, self.scan_times[-1]
        else:
            # Take the closest two times around t
            return idx - 1, self.scan_times[idx - 1], idx, self.scan_times[idx]

    def remove_orphan_lidars(self, pose_array):
        result_scan_times = []
        for timestamp in self.scan_times:
            pose_interpolated = pose_array.interpolated_pose_at_time(timestamp=timestamp)
            if pose_interpolated is None:
                continue
            # if a correct pose can be interpolated at time
            else:
                result_scan_times.append(timestamp)
        self.scan_times = result_scan_times

    def add_lidar_scans(self, keyframe_sampling=1):
        # First: add all keyframes with the known sampling
        for i in range(0, len(self.scan_times), keyframe_sampling):
            print("LidarScanArray: Adding Keyframe: ", i, "out of: ", len(self.scan_times), end='\r')
            self.add_lidar_scan(i)

    def add_lidar_scan(self, index):
        print('add_lidar_scan: adding LiDAR with scan_time: ', self.scan_times[index])
        kf = LiDARScan(directory=self.directory, scan_time=self.scan_times[index], parameters=self.parameters)
        self.lidar_scans.append(kf)

    # def load_pointclouds(self):
    #     for i in range(0, len(self.lidar_scans)):
    #         print("Keyframemanager: Loading Pointcloud: ", i, "out of: ", len(self.lidar_scans), end='\r')
    #         self.lidar_scans[i].load_pointcloud()

    def load_pointcloud(self, i):
        self.lidar_scans[i].load_pointcloud()

    def unload_pointcloud(self, i):
        self.lidar_scans[i].unload_pointcloud()

    def save_pointcloud(self, i):
        self.lidar_scans[i].save_pointcloud()

    def save_pointcloud_as_mesh(self, i):
        self.lidar_scans[i].save_pointcloud_as_mesh()

    def pre_process(self, index):
        self.lidar_scans[index].pre_process(method=self.method)

    def filter_points(self, index):
        self.lidar_scans[index].filter_points()

    def estimate_normals(self, index):
        self.lidar_scans[index].estimate_normals()

    def draw_cloud(self, index):
        self.lidar_scans[index].draw_cloud()

    # def draw_all_clouds(self):
    #     for i in range(len(self.lidar_scans)):
    #         self.lidar_scans[i].draw_cloud()

    # def visualize_keyframe(self, index):
    #     # self.keyframes[index].visualize_cloud()
    #     self.keyframes[index].draw_cloud()
    # def draw_cloud_visualizer(self, index):
    #     self.lidar_scans[index].draw_cloud_visualizer()

    def draw_all_clouds(self, sample=1):
        """
        Use o3d Visualizer to draw all clouds relative to the LiDAR reference frame.
        """
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        for i in range(0, len(self.scan_times), sample):
            vis.clear_geometries()
            self.load_pointcloud(i)
            self.lidar_scans[i].filter_height()
            self.lidar_scans[i].filter_radius()
            self.lidar_scans[i].down_sample()
            # view = vis.get_view_control()
            # view.set_up(np.array([1, 0, 0]))
            vis.add_geometry(self.lidar_scans[i].pointcloud, reset_bounding_box=True)
            # vis.update_geometry(self.lidar_scans[i].pointcloud)
            if not vis.poll_events():
                print("Window closed by user")
                break
            vis.update_renderer()
            time.sleep(0.01)
        vis.destroy_window()

    def draw_map(self, global_transforms, voxel_size=None, radii=None, heights=None, clear=False, keyframe_sampling=1):
        """
        Builds map rendering updates at each frame.

        Caution: the map is not built, but the o3d window is in charge of storing the points
        and viewing them.
        """
        print("VISUALIZING MAP FROM LIDARSCANS")
        print('NOW, BUILD THE MAP')

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        # transform all keyframes to global coordinates.
        # pointcloud_global = o3d.geometry.PointCloud()
        # caution, the visualizer only adds the transformed pointcloud to
        # the window, without removing the other geometries
        # the global map (pointcloud_global) is not built.
        for i in range(0, len(self.lidar_scans), keyframe_sampling):
            if clear:
                vis.clear_geometries()
            print("LiDAR scan: ", i, "out of: ", len(self.lidar_scans), end='\r')
            kf = self.lidar_scans[i]
            kf.load_pointcloud()
            kf.filter_radius(radii=radii)
            kf.filter_height(heights=heights)
            kf.down_sample(voxel_size=voxel_size)
            Ti = global_transforms[i]
            # forget about height in transform
            # if terraplanist:
            #     pi = Ti.pos()
            #     pi[2] = 0
            #     Ri = Ti.R()
            #     # Ri.array[0:3, 2] = np.array([0, 0, 1])
            #     Ti = HomogeneousMatrix(pi, Ri)
            # transform to global and
            pointcloud_temp = kf.transform(T=Ti.array)
            # unload pointcloud to save memroy
            kf.unload_pointcloud()
            # yuxtaponer los pointclouds
            # pointcloud_global = pointcloud_global + pointcloud_temp
            # vis.add_geometry(pointcloud_global, reset_bounding_box=True)
            vis.add_geometry(pointcloud_temp, reset_bounding_box=True)
            vis.get_render_option().point_size = 1
            # vis.update_geometry(pointcloud_global)
            vis.poll_events()
            vis.update_renderer()
        print('FINISHED! Use the window renderer to observe the map!')
        vis.run()
        vis.destroy_window()

    def build_map(self, global_transforms, keyframe_sampling=10, radii=None, heights=None, voxel_size=0.1):
        """
        Caution: in this case, the map is built using a pointcloud and adding the points to it. This may require a great
        amount of memory, however the result may be saved easily
        """
        if radii is None:
            radii = [0.5, 35.0]
        if heights is None:
            heights = [-120.0, 120.0]

        print('NOW, BUILD THE MAP')
        # transform all keyframes to global coordinates.
        pointcloud_global = o3d.geometry.PointCloud()
        for i in range(0, len(self.lidar_scans), keyframe_sampling):
            print("Keyframe: ", i, "out of: ", len(self.lidar_scans), end='\n')
            kf = self.lidar_scans[i]
            kf.load_pointcloud()
            kf.filter_radius(radii=radii)
            kf.filter_height(heights=heights)
            # print('Local pointcloud info')
            # print(kf.pointcloud)
            kf.down_sample(voxel_size=voxel_size)
            # print('Local pointcloud info')
            # print(kf.pointcloud)
            Ti = global_transforms[i]
            # transform to global and
            pointcloud_temp = kf.transform(T=Ti.array)
            # yuxtaponer los pointclouds
            pointcloud_global = pointcloud_global + pointcloud_temp
            # unload pointcloud to save memory
            kf.unload_pointcloud()
        print('FINISHED! Use the renderer to view the map')
        # draw the whole map
        o3d.visualization.draw_geometries([pointcloud_global])
        return pointcloud_global

    # def delete_empty_spaces_in_map_prob(self, pointcloud_global, global_transforms, keyframe_sampling=10, radii=None, heights=None, voxel_size=0.1):
    #     """
    #     Caution: in this case, the map is built using a pointcloud and adding the points to it. This may require a great
    #     amount of memory, however the result may be saved easily
    #     """
    #     if radii is None:
    #         radii = [0.5, 35.0]
    #     if heights is None:
    #         heights = [-120.0, 120.0]
    #     pointcloud_global_kdtree = o3d.geometry.KDTreeFlann(pointcloud_global)
    #     # paint global pointcloud
    #     # pointcloud_global.paint_uniform_color([0.1, 0.1, 0.8])
    #
    #     # Visualize the points to remove
    #     pointcloud_remove = o3d.geometry.PointCloud()
    #     # caution . radius
    #     # radius_remove = 0.2
    #     print(30*'=')
    #     print('NOW, DELETE EMPTY SPACES')
    #     print('FOR EACH POINTCLOUD, FIND THE POINTS THAT SHOULD BE IN EMPTY SPACES AND REMOVE THEM')
    #     print(30 * '=')
    #     # all_indices = set()
    #     all_indices = [] #set()
    #
    #     # for i in range(0, len(self.lidar_scans), keyframe_sampling):
    #     for i in range(12000, 12500, 5):
    #         print("Keyframe: ", i, "out of: ", len(self.lidar_scans), end='\n')
    #         kf = self.lidar_scans[i]
    #         kf.load_pointcloud()
    #         kf.filter_radius(radii=radii)
    #         kf.filter_height(heights=heights)
    #         kf.down_sample(voxel_size=voxel_size)
    #         Ti = global_transforms[i]
    #         # transform to global and
    #         pointcloud_temp = kf.transform(T=Ti.array)
    #         pi = Ti.pos()
    #         points_global_i = pointcloud_temp.points
    #         # remove local points between pi (origin) and points_global_i using spheres of radius r
    #         idxs = self.delete_empty_spaces_at_local(pointcloud_global_kdtree, pi, points_global_i)
    #         # pcd_dynamic = pointcloud_global.select_by_index(list(idxs), invert=False)
    #         # draw the whole map and the points to remove
    #         # o3d.visualization.draw_geometries([pcd_dynamic])
    #         # all_indices.update(idxs)
    #         all_indices.append(idxs)
    #         # unload pointcloud to save memory
    #         kf.unload_pointcloud()
    #     # Init all indices with a 100% occupancy likelihood
    #     indices_occupancy = 100.0*np.ones(len(pointcloud_global.points))
    #     # all_indices.update(idxs)
    #     all_indices = [item for sublist in all_indices for item in sublist]
    #
    #     # reduce the indices that should be in empty space
    #     for i in all_indices:
    #         indices_occupancy[i] -= 1.0
    #
    #     plt.plot(indices_occupancy)
    #     plt.show()
    #     indices_to_remove = np.where(indices_occupancy < 95.0)[0]
    #
    #     # removable_points = np.asarray(pointcloud_global.points)[all_indices]
    #     # print('Found ', len(removable_points), 'dynamic points on the whole process at the robot moving origin')
    #     # build a temporary removal pointcloud
    #     # pointcloud_remove.points = o3d.utility.Vector3dVector(removable_points)
    #     print('FINISHED! Use the renderer to view the map')
    #     pcd_dynamic = pointcloud_global.select_by_index(list(indices_to_remove), invert=False)
    #     pcd_dynamic.paint_uniform_color([1.0, 0, 0])
    #     o3d.visualization.draw_geometries([pcd_dynamic])
    #
    #     # remove the points and draw
    #     pcd_filtered = pointcloud_global.select_by_index(list(indices_to_remove), invert=True)
    #     # pcd_filtered.paint_uniform_color([1.0, 0, 0])
    #     o3d.visualization.draw_geometries([pcd_filtered])
    #     # draw the whole map
    #     # o3d.visualization.draw_geometries([pointcloud_global, pcd_filtered])
    #     o3d.visualization.draw_geometries([pointcloud_global, pcd_dynamic])
    #     return pointcloud_global

    def delete_empty_spaces_in_map(self, pointcloud_global, global_transforms, keyframe_sampling=10, radii=None,
                                   heights=None, voxel_size=0.2):
        """
        Caution: in this case, the map is built using a pointcloud and adding the points to it. This may require a great
        amount of memory, however the result may be saved easily
        """
        pointcloud_global_kdtree = o3d.geometry.KDTreeFlann(pointcloud_global)

        # Visualize the points to remove
        print(30*'=')
        print('NOW, DELETE EMPTY SPACES')
        print('FOR EACH POINTCLOUD, FIND THE POINTS THAT SHOULD BE IN EMPTY SPACES AND REMOVE THEM')
        print(30 * '=')
        all_indices = []
        # for i in range(0, len(self.lidar_scans), keyframe_sampling):
        for i in range(11050, 11100, 5):
            print("Keyframe: ", i, "out of: ", len(self.lidar_scans), end='\n')
            kf = self.lidar_scans[i]
            kf.load_pointcloud()
            kf.filter_radius(radii=radii)
            kf.filter_height(heights=heights)
            kf.down_sample(voxel_size=voxel_size)
            Ti = global_transforms[i]
            # transform to global and
            pointcloud_lidar = kf.transform(T=Ti.array)
            pi = Ti.pos()
            # points_lidar_global_i = pointcloud_lidar.points
            # remove local points between pi (origin) and points_global_i using spheres of radius r
            # idxs = self.delete_empty_spaces_at_local2(pointcloud_global_kdtree, pi, points_lidar_global_i)
            # idxs = self.delete_empty_spaces_at_local3(pointcloud_global, pointcloud_global_kdtree, pi, pointcloud_lidar)
            idxs = self.delete_empty_spaces_at_local4(pointcloud_global, pointcloud_global_kdtree, pi, pointcloud_lidar)
            pcd_dynamic = pointcloud_global.select_by_index(list(idxs), invert=False)
            # draw the whole map and the points to remove
            # o3d.visualization.draw_geometries([pcd_dynamic])
            all_indices.append(idxs)
            # unload pointcloud to save memory
            kf.unload_pointcloud()
        # Init all indices with a 100% occupancy likelihood
        indices_occupancy = 0.0*np.ones(len(pointcloud_global.points))
        all_indices = [item for sublist in all_indices for item in sublist]
        # reduce the indices that should be in empty space
        for i in all_indices:
            indices_occupancy[i] += 1.0

        plt.plot(indices_occupancy)
        plt.show()
        indices_to_remove = np.where(indices_occupancy > 2.0)[0]

        # removable_points = np.asarray(pointcloud_global.points)[all_indices]
        # print('Found ', len(removable_points), 'dynamic points on the whole process at the robot moving origin')
        # build a temporary removal pointcloud
        # pointcloud_remove.points = o3d.utility.Vector3dVector(removable_points)
        print('FINISHED! Use the renderer to view the map')
        pcd_dynamic = pointcloud_global.select_by_index(list(indices_to_remove), invert=False)
        pcd_dynamic.paint_uniform_color([1.0, 0, 0])
        o3d.visualization.draw_geometries([pcd_dynamic])
        o3d.visualization.draw_geometries([pointcloud_global, pcd_dynamic])

        # remove the points and draw
        pcd_filtered = pointcloud_global.select_by_index(list(indices_to_remove), invert=True)
        # pcd_filtered.paint_uniform_color([1.0, 0, 0])
        o3d.visualization.draw_geometries([pcd_filtered])
        # draw the whole map
        # o3d.visualization.draw_geometries([pointcloud_global, pcd_filtered])
        o3d.visualization.draw_geometries([pointcloud_global, pcd_dynamic])
        return pointcloud_global


    def delete_empty_spaces_at_local(self, pointcloud_global_kdtree, pi, points_global_i):
        # robot radius remove must be > than beam radius remove
        robot_radius_remove = 0.25
        beam_radius_remove = 0.02
        local_indices = [] # set()
        # remove at the robot center. use a radius that approximates the whole robot
        [_, idxs, _] = pointcloud_global_kdtree.search_radius_vector_3d(pi, robot_radius_remove)
        # local_indices.update(idxs)
        local_indices.append(list(idxs))
        # for each point in pointcloud, find interpolations
        for pj in points_global_i:
            # the distance from the LiDAR origin to the point
            dist = np.linalg.norm(pj - pi)
            if dist == 0:
                continue
            if dist <= robot_radius_remove:
                continue
            u = (pj - pi)/dist
            r1 = robot_radius_remove
            r2 = dist-10.0*beam_radius_remove
            n = int(np.ceil((r2-r1)/(2*beam_radius_remove)))
            r = np.linspace(r1, r2, n)
            for ri in r:
                # interpolate point using initial point, unit vector and interpolated distance
                pn = pi + ri*u
                # find indices close to the point
                [_, idxs, _] = pointcloud_global_kdtree.search_radius_vector_3d(pn, beam_radius_remove)
                # add to the list.
                # local_indices.update(idxs)
                local_indices.append(list(idxs))
        # flatten list
        local_indices = [item for sublist in local_indices for item in sublist]
        return local_indices


    def delete_empty_spaces_at_local2(self, pointcloud_global_kdtree, p0, points_global_i_lidar):
        # robot radius remove must be > than beam radius remove
        # two ways to remove.
        # a) Remove where the robot has been exactly
        # b) Remove on the very clear empty spaces with laser beams with distance over
        r_min = 20.0
        robot_radius_remove = 0.3
        beam_radius_remove = 0.05
        local_indices = []

        # remove radius below r_min in pointcloud
        dist = np.linalg.norm(points_global_i_lidar - p0, axis=1)
        idxs = np.where(dist > r_min)
        points_global_i_lidar = np.asarray(points_global_i_lidar)[idxs]

        # remove at the robot center. use a radius that approximates the whole robot
        [_, idxs, _] = pointcloud_global_kdtree.search_radius_vector_3d(p0, robot_radius_remove)
        # local_indices.update(idxs)
        local_indices.append(list(idxs))
        # for each point in pointcloud, find interpolations
        for pj in points_global_i_lidar:
            # the actual distance from the LiDAR origin to the point
            dist = np.linalg.norm(pj - p0)
            if dist == 0:
                continue
            if dist <= robot_radius_remove:
                continue
            if dist <= r_min:
                continue
            u = (pj - p0)/dist
            r1 = robot_radius_remove
            r2 = dist/5.0 #yes, find points that should be empty at a distance below a third-10.0*beam_radius_remove
            n = int(np.ceil((r2-r1)/(2*beam_radius_remove)))
            r = np.linspace(r1, r2, n)
            for ri in r:
                # interpolate point using initial point, unit vector and interpolated distance
                pn = p0 + ri*u
                # find indices close to the point
                [_, idxs, _] = pointcloud_global_kdtree.search_radius_vector_3d(pn, beam_radius_remove)
                # add to the list.
                local_indices.append(list(idxs))
        # flatten list
        local_indices = [item for sublist in local_indices for item in sublist]
        return local_indices



    def delete_empty_spaces_at_local3(self, pointcloud_global, pointcloud_global_kdtree, p0, pointcloud_lidar):
        """
        Obtain a submap of points around the robot position p0. We are willing to detect dynamic objects at this place.
        We are obtaining points that should be at a larger distance, according to the lidar observation
        """
        # robot radius remove must be > than beam radius remove
        # two ways to remove.
        # a) Remove where the robot has been exactly
        # b) Remove on the very clear empty spaces with laser beams with distance over
        global_submap_radius = 5.0

        # min radius of the lidar to check for empty spaces, i. e. 2*global_submap_radius
        r_min = 10.0
        # remove points around the robot
        robot_radius_remove = 0.3
        # detect points at this distance from the beam.
        beam_radius_remove = 0.1
        # the total list of indices to remove from the global map
        local_indices = []

        # the initial maps
        pointcloud_lidar.paint_uniform_color([1.0, 0.0, 0.0])
        o3d.visualization.draw_geometries([pointcloud_global, pointcloud_lidar])


        # remove at the robot center. use a radius that approximates the whole robot
        [_, idxs, _] = pointcloud_global_kdtree.search_radius_vector_3d(p0, robot_radius_remove)
        local_indices.append(list(idxs))


        # OBTAIN A SUBMAP OF THE GLOBAL MAP
        # also filter points above and below z from p0!!!
        [_, idxs, _] = pointcloud_global_kdtree.search_radius_vector_3d(p0, global_submap_radius)
        pointcloud_global = pointcloud_global.select_by_index(list(idxs), invert=False)
        # filter z
        pointcloud_global_points = np.asarray(pointcloud_global.points)
        mask = (pointcloud_global_points[:, 2] - p0[2] > -0.5) & (pointcloud_global_points[:, 2] - p0[2] < 1.5)
        idxs = np.where(mask)[0]
        pointcloud_global = pointcloud_global.select_by_index(list(idxs), invert=False)
        o3d.visualization.draw_geometries([pointcloud_global, pointcloud_lidar])
        pointcloud_global_points = np.asarray(pointcloud_global.points)

        # FILTER LIDAR points
        pointcloud_lidar_points = np.asarray(pointcloud_lidar.points)
        # obtain points/radius that are above r_min
        dist = np.linalg.norm(pointcloud_lidar_points - p0, axis=1)
        idxs = np.where(dist > r_min)[0]
        pointcloud_lidar = pointcloud_lidar.select_by_index(list(idxs), invert=False)
        pointcloud_lidar_points = np.asarray(pointcloud_lidar.points)
        o3d.visualization.draw_geometries([pointcloud_global, pointcloud_lidar])



        print('Number of points in lidar: ', len(pointcloud_lidar_points))
        print('Number of points in submap: ', len(pointcloud_global_points))


        # for each point in the lidar pointcloud LiDAR observation find points at close distance
        # per a cadascú dels rajos del lidar no hem de trobar cap objecte més a prop
        # esta observació del lidar ha sigut ja filtrada
        i = 0
        for pj in pointcloud_lidar_points:
            i += 1
            print('i out of: ', i, len(pointcloud_lidar_points))
            # the actual distance from the LiDAR origin to the point
            dist = np.linalg.norm(pj - p0)
            # if dist == 0:
            #     continue
            # if dist <= robot_radius_remove:
            #     continue
            # if dist <= r_min:
            #     continue
            # compute the unit direction vector
            u = (pj - p0)/dist
            p_rel = pointcloud_global_points - p0
            proj = np.dot(p_rel, u)[:, None]*u
            dist_vectors = p_rel - proj
            distances = np.linalg.norm(dist_vectors, axis=1)
            # find distances to the ray which are below threshold
            idxs = np.where(distances < beam_radius_remove)[0]
            # close_points_to_line = pointcloud_global[mask]
            # store the original indices of the global map
            # if len(idxs) > 5:
            #     continue
            idxs = candidates_local[idxs]
            local_indices.append(list(idxs))

        # flatten list
        local_indices = [item for sublist in local_indices for item in sublist]
        return local_indices


    def delete_empty_spaces_at_local4(self, pointcloud_global, pointcloud_global_kdtree, p0, pointcloud_lidar):
        """
        Obtain a submap of points around the robot position p0. We are willing to detect dynamic objects at this place.
        We are obtaining points that should be at a larger distance, according to the lidar observation
        """
        # robot radius remove must be > than beam radius remove
        # two ways to remove.
        # a) Remove where the robot has been exactly
        # b) Remove on the very clear empty spaces with laser beams with distance over
        global_submap_radius = 5.0

        # min radius of the lidar to check for empty spaces, i. e. 2*global_submap_radius
        r_min = 10.0
        # remove points around the robot
        robot_radius_remove = 0.3
        # detect points at this distance from the beam.
        beam_radius_remove = 0.01
        # the total list of indices to remove from the global map
        local_indices = []


        # remove at the robot center. use a radius that approximates the whole robot
        [_, idxs, _] = pointcloud_global_kdtree.search_radius_vector_3d(p0, robot_radius_remove)
        local_indices.append(list(idxs))

        pointcloud_global_points = np.asarray(pointcloud_global.points)

        # OBTAIN A SUBMAP OF THE GLOBAL MAP
        # also filter points above and below z from p0!!!
        [_, indices_radius, _] = pointcloud_global_kdtree.search_radius_vector_3d(p0, global_submap_radius)
        mask = np.zeros(len(pointcloud_global_points), dtype=bool)
        mask[indices_radius] = True
        mask = mask & (pointcloud_global_points[:, 2] - p0[2] > -0.8) & (pointcloud_global_points[:, 2] - p0[2] < 1.5)
        indices_in_global_map = np.where(mask)[0]
        pointcloud_global = pointcloud_global.select_by_index(list(indices_in_global_map), invert=False)
        pointcloud_lidar.paint_uniform_color([1.0, 0.1, 0.1])
        o3d.visualization.draw_geometries([pointcloud_global, pointcloud_lidar])
        pointcloud_global_points = np.asarray(pointcloud_global.points)
        pointcloud_lidar_points = np.asarray(pointcloud_lidar.points)

        print('Number of points in lidar: ', len(pointcloud_lidar_points))
        print('Number of points in submap: ', len(pointcloud_global_points))

        existence_prob = 0.1*np.ones(len(indices_in_global_map))
        # for each point in the lidar pointcloud LiDAR observation find points at close distance
        # per a cadascú dels rajos del lidar no hem de trobar cap objecte més a prop
        # esta observació del lidar ha sigut ja filtrada
        # i = 0
        for i in range(len(pointcloud_global_points)):
            # i += 1
            # print('i out of: ', i, len(pointcloud_lidar_points))
            # the actual distance from the LiDAR origin to the point
            pj = pointcloud_global_points[i]
            dist = np.linalg.norm(pointcloud_lidar_points - pj, axis=1)
            # find distances to the ray which are below threshold
            # idxs = np.where(dist < beam_radius_remove)[0]
            d = min(dist)
            if d < 0.15:
                existence_prob[i] += 0.1
            else:
                existence_prob[i] -= 0.1

        indices_in_global_map_exist = np.where(existence_prob > 0.0)[0]

        indices_in_global_map_no_exist = np.where(existence_prob < 0.0)[0]


            # # close_points_to_line = pointcloud_global[mask]
            # # store the original indices of the global map
            # # if len(idxs) > 5:
            # #     continue
            # # switch to global indices
            # idxs = indices_in_global_map[idxs]
            # local_indices.append(list(idxs))
        # for pj in pointcloud_lidar_points:
        #     i += 1
        #     # print('i out of: ', i, len(pointcloud_lidar_points))
        #     # the actual distance from the LiDAR origin to the point
        #     dist = np.linalg.norm(pj - p0)
        #     # compute the unit direction vector
        #     u = (pj - p0) / dist
        #     p_rel = pointcloud_global_points - p0
        #     proj = np.dot(p_rel, u)[:, None] * u
        #     dist_vectors = p_rel - proj
        #     distances = np.linalg.norm(dist_vectors, axis=1)
        #     # find distances to the ray which are below threshold
        #     idxs = np.where(distances < beam_radius_remove)[0]
        #
        #     # close_points_to_line = pointcloud_global[mask]
        #     # store the original indices of the global map
        #     # if len(idxs) > 5:
        #     #     continue
        #     # switch to global indices
        #     idxs = indices_in_global_map[idxs]
        #     local_indices.append(list(idxs))

        # flatten list
        local_indices = [item for sublist in local_indices for item in sublist]
        return local_indices


        #
        # # filter z
        # pointcloud_global_points = np.asarray(pointcloud_global.points)
        # mask = (pointcloud_global_points[:, 2] - p0[2] > -0.5) & (pointcloud_global_points[:, 2] - p0[2] < 1.5)
        # idxs = np.where(mask)[0]
        # pointcloud_global = pointcloud_global.select_by_index(list(idxs), invert=False)
        # o3d.visualization.draw_geometries([pointcloud_global, pointcloud_lidar])
        # # pointcloud_global_points = np.asarray(pointcloud_global.points)
        #
        # # FILTER LIDAR points
        # pointcloud_lidar_points = np.asarray(pointcloud_lidar.points)
        # # obtain points/radius that are above r_min
        # dist = np.linalg.norm(pointcloud_lidar_points - p0, axis=1)
        # idxs = np.where(dist > r_min)[0]
        # pointcloud_lidar = pointcloud_lidar.select_by_index(list(idxs), invert=False)
        # pointcloud_lidar_points = np.asarray(pointcloud_lidar.points)
        # o3d.visualization.draw_geometries([pointcloud_global, pointcloud_lidar])









