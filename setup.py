from setuptools import setup, find_packages

setup(
    name='simbanator',
    version='0.2.0',
    packages=find_packages(include=['simbanator', 'simbanator.*']),
    include_package_data=True,
    install_requires=[
        'numpy',
        'scipy',
        'astropy',
        'matplotlib',
        'h5py',
        'unyt',
        'Pillow',
        'psutil',
    ],
)

