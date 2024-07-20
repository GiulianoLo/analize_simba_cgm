from setuptools import setup, find_packages

setup(
    name='analize_simba_cgm',
    version='0.1.0',
    packages=find_packages(where='analize_simba_cgm_code'),
    package_dir={'': 'analize_simba_cgm_code'},
    include_package_data=False,
    install_requires=[
        'numpy',
        'scipy',
        'astropy',
        'matplotlib',
    ],
    entry_points={
        'console_scripts': [
            ,
        ],
    },
)

