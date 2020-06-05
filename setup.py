import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="heritage-connector-pkg", 
    version="0.0.1",
    author="Science Musuem Group",
    description="Science Musuem Group, Heritage Connector",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TheScienceMuseum/heritage-connector",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)