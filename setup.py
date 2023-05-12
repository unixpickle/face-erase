from setuptools import setup

setup(
    name="face-erase",
    packages=[
        "face_erase",
    ],
    install_requires=[
        "torch",
        "Pillow",
    ],
    author="OpenAI",
)
