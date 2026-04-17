"""
launch_pata.py
==============
Lanza:
  1. robot_state_publisher  — publica TF desde el URDF de la_pata_sola
  2. joint_state_publisher_gui — sliders para mover las articulaciones
  3. rviz2                  — visualización 3D

Estructura esperada del paquete parcial2:
    parcial2/
    ├── launch/
    │   └── launch_pata.py          ← este archivo
    ├── urdf/
    │   └── la_pata_sola.urdf
    ├── meshes/
    │   ├── base_link.STL
    │   ├── link_c.STL
    │   ├── link_p.STL
    │   └── link_r.STL
    └── config/
        └── pata.rviz               (opcional, se crea automático si no existe)

Uso:
    colcon build --symlink-install
    source install/setup.bash
    ros2 launch parcial2 launch_pata.py
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("parcial2")

    urdf_path = os.path.join(pkg_share, "urdf", "la_pata_sola.urdf")
    rviz_path = os.path.join(pkg_share, "config", "pata.rviz")

    # Leer URDF como string para robot_state_publisher
    with open(urdf_path, "r") as f:
        robot_description = f.read()

    # ── robot_state_publisher ─────────────────────────────────────
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

    # ── joint_state_publisher_gui ─────────────────────────────────
    # Proporciona sliders interactivos para mover joint_c, joint_p, joint_r
    jsp_gui_node = Node(
        package="joint_state_publisher_gui",
        executable="joint_state_publisher_gui",
        name="joint_state_publisher_gui",
        output="screen",
    )

    # ── RViz2 ────────────────────────────────────────────────────
    rviz_args = ["-d", rviz_path] if os.path.exists(rviz_path) else []

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=rviz_args,
    )

    return LaunchDescription([
        rsp_node,
        jsp_gui_node,
        rviz_node,
    ])
