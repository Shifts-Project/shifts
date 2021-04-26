from setuptools import find_packages, setup


setup(
    name='ysdc_dataset_api',
    packages=find_packages(),
    install_requires=[
        "jupyter",
        "matplotlib",
        "numpy",
        "opencv-contrib-python-headless",
        "protobuf>=3.12.2",
        "torch>=1.5.0,<2.0.0",
        "torchvision>=0.6.0,<1.0.0",
        "transforms3d",
    ]
)