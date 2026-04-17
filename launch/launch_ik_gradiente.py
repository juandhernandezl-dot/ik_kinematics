"""
launch_ik_gradiente.py
==============
Lanza:
  1. robot_state_publisher  — publica TF desde LA_PATA_SOLA.urdf
  2. gui_pata               — GUI FK + IK numérica (Jacobiana)
  3. rviz2                  — visualización 3D

Uso:
    cd ~/parcial2_ws
    colcon build --symlink-install
    source install/setup.bash
    ros2 launch parcial2 launch_ik_gradiente.py
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("parcial2")

    urdf_path = os.path.join(pkg_share, "urdf", "LA_PATA_SOLA.urdf")
    rviz_path = os.path.join(pkg_share, "config", "pata.rviz")

    with open(urdf_path, "r") as f:
        robot_description = f.read()

    # ── robot_state_publisher ──────────────────────────────────────
    rsp_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            {"robot_description": robot_description},
            {"publish_frequency": 50.0},
        ],
    )

    # ── GUI FK + IK numérica ───────────────────────────────────────
    # Publica /joint_states → RSP genera TF → RViz2 mueve el modelo
    gui_node = Node(
        package="parcial2",
        executable="gui_pata",
        name="gui_pata",
        output="screen",
        emulate_tty=True,
    )

    # ── RViz2 ─────────────────────────────────────────────────────
    rviz_args = ["-d", rviz_path] if os.path.exists(rviz_path) else []
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=rviz_args,
    )

    return LaunchDescription([rsp_node, gui_node, rviz_node])
