#!/bin/bash 
file="$1"
source /home/airlab/ros2_ws/install/setup.bash
ros2 run airlab_lidar_3dgs pose_estimator "$file"
wait
ros2 run airlab_lidar_3dgs registration "$file" --diag
wait
ros2 run airlab_lidar_3dgs reconstruction "$file"
wait
ros2 run airlab_lidar_3dgs export_colmap "$file" --diag