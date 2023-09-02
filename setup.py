from setuptools import setup

setup(
    name="face-erase",
    packages=[
        "face_erase",
    ],
    include_package_data=True,
    package_data={
        "face_erase": [
            "binary/yolov8n-face-640x640.torchscript",
        ],
    },
    install_requires=[
        "Pillow",
        "torch",
        "torchvision",
        "tqdm",
    ],
    author="Alex Nichol",
    version="0.1.0",
)
