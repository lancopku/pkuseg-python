import setuptools

#with open("README.md", "r") as fh:
#    long_description = fh.read()
long_description='PKUSeg-python'

setuptools.setup(
        name="PKUSeg",
        version="0.0.4",
        author="Lanco",
        author_email="luoruixuan97@pku.edu.cn",
        description="A small package for Chinese word segmentation",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/lancopku/PKUSeg-python",
        packages=setuptools.find_packages(),
        package_data={'': ['*.txt*']},
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            ],
        )
