from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'parcial2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # Registro del paquete en ament (obligatorio)
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        # Archivos de launch
        ('share/' + package_name + '/launch',
            glob('launch/*.py')),

        # URDF
        ('share/' + package_name + '/urdf',
            glob('urdf/*')),

        # Meshes STL
        ('share/' + package_name + '/meshes',
            glob('meshes/*')),

        # Config RViz (opcional, incluye todo lo que haya en config/)
        ('share/' + package_name + '/config',
            glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='reds7',
    maintainer_email='juan_d.hernandez_l@uao.edu.co',
    description='Parcial 2 — visualización y cinemática de LA_PATA_SOLA (DH, 3 DOF + efector fijo)',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Aquí se agregarán los ejecutables en siguientes pasos
            # Ejemplo:  'gui_pata = parcial2.gui_pata:main',
        ],
    },
)
