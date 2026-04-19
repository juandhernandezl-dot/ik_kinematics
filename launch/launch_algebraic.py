#!/usr/bin/env python3
"""
launch_algebraic.py — Launch para cinemática inversa analítica de LA_PATA_SOLA
===============================================================================

Este fichero de lanzamiento inicia tres nodos:

1. **robot_state_publisher** para publicar el URDF de LA_PATA_SOLA en el
   sistema de transformadas.  Lee el fichero URDF directamente desde el
   paquete instalado y lo suministra como parámetro `robot_description`.

2. **gui_algebraic** que es el nodo PyQt5 implementado en `gui_algebraic.py`.
   Este nodo resuelve la cinemática inversa utilizando un método analítico
   adaptado a la geometría de la pata y publica los estados articulares
   necesarios en `/joint_states`.

3. **rviz2** para visualizar la pata y los ejes en 3D.  Si existe un
   fichero de configuración RViz bajo `config/pata.rviz` se utiliza; de lo
   contrario, RViz se lanza con su configuración por defecto.

Uso:
    ros2 launch parcial2 launch_algebraic.py
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory('parcial2')
    # Rutas de recursos
    urdf_path = os.path.join(pkg_share, 'urdf', 'LA_PATA_SOLA.urdf')
    rviz_path = os.path.join(pkg_share, 'config', 'pata.rviz')
    # Leer el URDF a cadena
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
    # Nodo de la GUI analítica
    gui_node = Node(
        package='parcial2',
        executable='gui_algebraic',
        name='gui_algebraic',
        output='screen',
        emulate_tty=True,
    )
    # Nodo RViz2 (usar configuración si existe)
    rviz_args = ['-d', rviz_path] if os.path.exists(rviz_path) else []
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=rviz_args,
    )
    return LaunchDescription([rsp_node, gui_node, rviz_node])