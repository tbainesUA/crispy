# import setuptools

# setuptools.setup(
#     name="crispy",
#     version="0.0.1",
#     author="Example Author",
#     author_email="author@example.com",
#     description="A small example package",

#     url="https://github.com/pypa/sampleproject",
#     classifiers=[
#         "Programming Language :: Python :: 3",
#         "License :: OSI Approved :: MIT License",
#         "Operating System :: OS Independent",
#     ],
#     # package_dir={"crispy": "crispy"},
#     packages=setuptools.find_packages(),
#     python_requires=">=3.6",
# )

#!/usr/bin/env python

#!/usr/bin/env python

from setuptools import setup, find_packages
# setup(
#     name='crispy',
#     version="1.0.0", 
#     author='Tyler Baines',
#     author_email = 'tyler.baines@nasa.gov',
#     url = 'https://github.com/tbainesUA/crispy',
#     packages =['crispy'],
#     license = ['GNU GPLv3'],
#     description ='The Coronagraph and Rapid Imaging Spectrograph in Python',
#     include_package_data=True,
#     classifiers = [
#         'Development Status :: 3 - Alpha',#5 - Production/Stable',
#         'Intended Audience :: Science/Research',
#         'Topic :: Scientific/Engineering',
#         'Programming Language :: Python'
#         ],
#     package_dir={"crispy":'crispy'}

# )
setup(
    name='crispy',
    version="0.9", 
    author='Maxime Rizzo',
    author_email = 'maxime.j.rizzo@nasa.gov',
    url = 'https://github.com/mjrfringes/crispy',
    packages = ['crispy'],
    license = ['GNU GPLv3'],
    description ='The Coronagraph and Rapid Imaging Spectrograph in Python',
    package_dir = {"crispy":'crispy'},
    include_package_data=True,
    classifiers = [
        'Development Status :: 3 - Alpha',#5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python'
        ],
)