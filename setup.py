from setuptools import find_packages, setup
from glob import glob

package_name = 'parcial2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch',    glob('launch/*.py')),
        ('share/' + package_name + '/urdf',      glob('urdf/*')),
        ('share/' + package_name + '/meshes',    glob('meshes/*')),
        ('share/' + package_name + '/config',    glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='reds7',
    maintainer_email='juan_d.hernandez_l@uao.edu.co',
    description='Parcial 2 — FK + IK numérica para LA_PATA_SOLA (3 DOF + efector fijo)',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # GUI FK + IK numérica (Jacobiana) para LA_PATA_SOLA
            'gui_pata = parcial2.gui_pata:main',
        ],
    },
)
