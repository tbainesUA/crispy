from setuptools import setup

setup(
    name='crispy',
    version="0.9", 
    author='Maxime Rizzo',
    author_email = 'maxime.j.rizzo@nasa.gov',
    url = 'https://github.com/mjrfringes/crispy',
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

