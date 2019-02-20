import setuptools
from distutils.extension import Extension

import numpy as np
from Cython.Build import cythonize

def setup_package():

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

    setuptools.setup(
        name="pkuseg",
        version="0.0.16",
        author="Lanco",
        author_email="luoruixuan97@pku.edu.cn",
        description="A small package for Chinese word segmentation",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/lancopku/pkuseg-python",
        packages=setuptools.find_packages(),
        package_data={"": ["*.txt*", "*.pkl", "*.npz"]},
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: Other/Proprietary License",
            "Operating System :: OS Independent",
        ],
        install_requires=["numpy>=1.16.0"],
        ext_modules=cythonize(extensions, annotate=True),
        zip_safe=False,
    )


if __name__ == "__main__":
    setup_package()
