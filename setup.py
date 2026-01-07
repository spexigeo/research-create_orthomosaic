from setuptools import setup, find_packages

setup(
    name="research-create-orthomosaic",
    version="0.1.0",
    description="Tiepoint matcher for orthomosaic creation from drone imagery",
    packages=find_packages(),
    install_requires=[
        "h3>=3.7.0",
        "numpy>=1.24.0",
        "networkx>=3.0",
        "matplotlib>=3.7.0",
        "pillow>=10.0.0",
        "torch>=2.0.0",
        "lightglue>=0.1.0",
        "opencv-python>=4.8.0",
        "tqdm>=4.65.0",
    ],
    python_requires=">=3.8",
)
