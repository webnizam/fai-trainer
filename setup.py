from setuptools import setup, find_packages

setup(
    name="fai-trainer",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "Pillow",
        "matplotlib",
        "tqdm",
        "numpy",
        "scipy",
    ],
    entry_points={
        "console_scripts": [
            "fai-trainer=trainer.main:main",
        ],
    },
)
