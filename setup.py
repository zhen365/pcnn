import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pcnn",
    version="0.1",
    author="lidezhen",
    author_email="lidezhenw@163.com",
    description="A toolbox of Pulse Coupled Neural Network (PCNN) for image processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lidezhenw/pcnn",
    license="GPLv3",
    install_requires=["numpy", "scipy", "opencv-python"],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
  ],
)