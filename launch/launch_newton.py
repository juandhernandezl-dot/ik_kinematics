#!/usr/bin/env python3
"""
launch_newton.py — Launch para cinemática inversa por método de Newton de LA_PATA_SOLA
=======================================================================================

Este lanzamiento arranca:

1. **robot_state_publisher** para publicar las transformaciones del URDF.

2. **gui_newton**: la interfaz gráfica implementada en `gui_newton.py` que
   utiliza el método numérico de Newton para resolver la cinemática inversa.
   La GUI también dispone de una pestaña de cinemática directa para
   visualización y ajuste manual de los ángulos.

3. **rviz2** para visualizar el robot y las transformaciones.  Si está
   disponible un fichero `pata.rviz` en el directorio de configuración del
   paquete, se utiliza como configuración predeterminada.

Uso:
    ros2 launch parcial2 launch_newton.py
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory('parcial2')
    urdf_path = os.path.join(pkg_share, 'urdf', 'LA_PATA_SOLA.urdf')
    rviz_path = os.path.join(pkg_share, 'config', 'pata.rviz')
    # Leer URDF
    with open(urdf_path, 'r') as f:
        robot_description = f.read()
    rsp_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description}],
    )
    gui_node = Node(
        package='parcial2',
        executable='gui_newton',
        name='gui_newton',
        output='screen',
        emulate_tty=True,
    )
    rviz_args = ['-d', rviz_path] if os.path.exists(rviz_path) else []
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=rviz_args,
    )
    return LaunchDescription([rsp_node, gui_node, rviz_node])