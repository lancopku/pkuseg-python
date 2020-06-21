import setuptools
import os
from distutils.extension import Extension

import numpy as np

def is_source_release(path):
    return os.path.exists(os.path.join(path, "PKG-INFO"))

def setup_package():
    root = os.path.abspath(os.path.dirname(__file__))

    long_description = "pkuseg-python"

    extensions = [
        Extension(
            "pkuseg.inference",
            ["pkuseg/inference.pyx"],
            include_dirs=[np.get_include()],
            language="c++"
        ),
        Extension(
            "pkuseg.feature_extractor",
            ["pkuseg/feature_extractor.pyx"],
            include_dirs=[np.get_include()],
        ),
        Extension(
            "pkuseg.postag.feature_extractor",
            ["pkuseg/postag/feature_extractor.pyx"],
            include_dirs=[np.get_include()],
        ),
    ]
    
    if not is_source_release(root):
        from Cython.Build import cythonize
        extensions = cythonize(extensions, annotate=True)


    setuptools.setup(
        name="pkuseg",
        version="0.0.25",
        author="Lanco",
        author_email="luoruixuan97@pku.edu.cn",
        description="A small package for Chinese word segmentation",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/lancopku/pkuseg-python",
        packages=setuptools.find_packages(),
        package_data={"": ["*.txt*", "*.pkl", "*.npz", "*.pyx", "*.pxd"]},
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: Other/Proprietary License",
            "Operating System :: OS Independent",
        ],
        install_requires=["cython", "numpy>=1.16.0"],
        setup_requires=["cython", "numpy>=1.16.0"],
        ext_modules=extensions,
        zip_safe=False,
    )


if __name__ == "__main__":
    setup_package()
