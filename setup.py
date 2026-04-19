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
        ('share/' + package_name + '/launch',  glob('launch/*.py')),
        ('share/' + package_name + '/urdf',    glob('urdf/*')),
        ('share/' + package_name + '/meshes',  glob('meshes/*')),
        ('share/' + package_name + '/config',  glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='reds7',
    maintainer_email='juan_d.hernandez_l@uao.edu.co',
    description='Parcial 2 — FK/IK para LA_PATA_SOLA (DH/MTH)',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gui_pata          = parcial2.gui_pata:main',
            'gui_mth           = parcial2.gui_mth:main',
            'gui_geometric     = parcial2.gui_geometric:main',
            'gui_algebraic     = parcial2.gui_algebraic:main',
            'gui_newton        = parcial2.gui_newton:main',
            'gui_gradient      = parcial2.gui_gradient:main',
            'gui_unificada     = parcial2.gui_unificada:main',
        ],
    },
)
