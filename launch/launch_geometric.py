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

    gui_node = Node(
        package="parcial2",
        executable="gui_geometric",
        name="gui_geometric",
        output="screen",
        emulate_tty=True,
    )

    rviz_args = ["-d", rviz_path] if os.path.exists(rviz_path) else []
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=rviz_args,
    )

    return LaunchDescription([rsp_node, gui_node, rviz_node])