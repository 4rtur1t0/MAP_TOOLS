# copiar kiss_slam.yaml
cd  ~/Applications/venv/bin
#CONFIG_FILE=/media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/kiss_slam.yaml
#EXPERIMENT=IO3-2025-06-16-13-49-28
#./kiss_slam_pipeline --config $CONFIG_FILE --visualize /media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/$EXPERIMENT/robot0/lidar/data

CONFIG_FILE=/media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/kiss_slam.yaml
EXPERIMENT=IO4-2025-06-16-15-56-11
./kiss_slam_pipeline --config $CONFIG_FILE --visualize /media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/$EXPERIMENT/robot0/lidar/data

CONFIG_FILE=/media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/kiss_slam.yaml
EXPERIMENT=IO5-2025-06-16-17-53-54
./kiss_slam_pipeline --config $CONFIG_FILE --visualize /media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/$EXPERIMENT/robot0/lidar/data


DO not visualize: remove --visualize
./kiss_slam_pipeline --config $CONFIG_FILE /media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/$EXPERIMENT/robot0/lidar/data
