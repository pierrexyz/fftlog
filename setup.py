from setuptools import setup

setup(
    name='fftlog',
    version='0.1.0',
    description='fftlog',
    author="Pierre Zhang",
    license='MIT',
    packages=['fftlog'],
    install_requires=['numpy', 'scipy'],
    package_dir = {'fftlog': 'fftlog'},
    zip_safe=False,

    classifiers = [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Environment :: Console",
        "Programming Language :: Python",
    ],
)
