import setuptools

#with open("README.md", "r") as fh:
#    long_description = fh.read()
long_description='pkuseg-python'

setuptools.setup(
        name="pkuseg",
        version="0.0.10",
        author="Lanco",
        author_email="luoruixuan97@pku.edu.cn",
        description="A small package for Chinese word segmentation",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/lancopku/pkuseg-python",
        packages=setuptools.find_packages(),
        package_data={'': ['*.txt*']},
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: Other/Proprietary License",
            "Operating System :: OS Independent",
            ],
        install_requires=[
            'numpy'
            ]
        )
