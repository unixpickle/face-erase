from setuptools import setup

setup(
    name="face-erase",
    packages=[
        "face_erase",
    ],
    install_requires=[
        "Pillow",
        "torch",
        "torchvision",
        "tqdm",
    ],
    author="OpenAI",
)
