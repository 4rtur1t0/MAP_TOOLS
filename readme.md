Simple tools to create:
a .pcd map joining all LiDAR scans along with their estimated
poses.

an occupancy grid map with a simple occupancy-grid algorithm
from LiDAR measurmentes.



STEPS NEEDED TO CREATE A MAP
- Use extract_rosbag to extract data from a Rosbag file to the EUROC/ASL format.
- use kiss_slam to estimate the path.
- with the path known (in tum format), RUN:
run_map_builder_tum.py, which can: visualize the global map,
  build the global .pcd file an write to disk, build an
  .png occupancy grid.