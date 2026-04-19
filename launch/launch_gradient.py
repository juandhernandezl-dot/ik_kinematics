#!/usr/bin/env python3
"""
launch_gradient.py — Launch para cinemática inversa por método del Gradiente de LA_PATA_SOLA
============================================================================================

Este archivo de lanzamiento arranca tres nodos esenciales para trabajar con
el método del gradiente aplicado a la pata del robot **LA_PATA_SOLA**:

1. **robot_state_publisher**: publica la descripción cinemática del robot en
   forma de transformaciones TF leyendo directamente el URDF del paquete.
2. **gui_gradient**: la interfaz gráfica (PyQt5) implementada en
   ``gui_gradient.py`` que resuelve la cinemática inversa mediante el método
   del gradiente numérico.  Esta GUI también permite explorar la
   cinemática directa y visualizar las matrices de transformación.
3. **rviz2**: visor 3D para mostrar la pata, los marcos de referencia y
   verificar en tiempo real la resolución de IK.  Si existe un fichero
   ``pata.rviz`` en el subdirectorio ``config`` del paquete, se utiliza
   como configuración predeterminada; de lo contrario, RViz se lanza con
   la configuración por defecto.

Uso:
    ros2 launch parcial2 launch_gradient.py

"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory('parcial2')
    urdf_path = os.path.join(pkg_share, 'urdf', 'LA_PATA_SOLA.urdf')
    rviz_path = os.path.join(pkg_share, 'config', 'pata.rviz')
    # Leer URDF como cadena
    with open(urdf_path, 'r') as f:
        robot_description = f.read()
    # Nodo robot_state_publisher
    rsp_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description}],
    )
    # Nodo de la GUI del gradiente
    gui_node = Node(
        package='parcial2',
        executable='gui_gradient',
        name='gui_gradient',
        output='screen',
        emulate_tty=True,
    )
    # Nodo RViz2 con configuración opcional
    rviz_args = ['-d', rviz_path] if os.path.exists(rviz_path) else []
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=rviz_args,
    )
    return LaunchDescription([rsp_node, gui_node, rviz_node])
