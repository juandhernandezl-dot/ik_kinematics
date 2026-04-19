#!/usr/bin/env python3
"""
launch_unificada.py — Lanza gui_unificada + RSP + RViz2

Inicia:
  1. robot_state_publisher  (TF desde el URDF LA_PATA_SOLA)
  2. gui_unificada          (FK + 5 métodos IK + Comparación)
  3. rviz2                  (visualización 3D)

Uso:
    ros2 launch parcial2 launch_unificada.py
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg  = get_package_share_directory("parcial2")
    urdf = os.path.join(pkg, "urdf", "LA_PATA_SOLA.urdf")
    rviz = os.path.join(pkg, "config", "pata.rviz")

    with open(urdf) as f:
        robot_description = f.read()

    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            {"robot_description": robot_description},
            {"publish_frequency": 50.0},
        ],
    )

    gui = Node(
        package="parcial2",
        executable="gui_unificada",
        name="gui_unificada",
        output="screen",
        emulate_tty=True,
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz] if os.path.exists(rviz) else [],
    )

    return LaunchDescription([rsp, gui, rviz_node])
