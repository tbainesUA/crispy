from setuptools import setup

setup(
    name='crispy',
    version="1.0.0", 
    author='Tyler Baines',
    author_email = 'tyler.baines@nasa.gov',
    url = 'https://github.com/tbainesUA/crispy',
    packages =['crispy'],
    license = ['GNU GPLv3'],
    description ='The Coronagraph and Rapid Imaging Spectrograph in Python',
    include_package_data=True,
    classifiers = [
        'Development Status :: 3 - Alpha',#5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python'
        ],

)

