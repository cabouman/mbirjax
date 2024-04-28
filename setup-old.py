from setuptools import setup, find_packages

NAME = "MBIRJAX"
VERSION = "0.1"
DESCR = "Project for creating a JAX version of MBIR"
REQUIRES = ['numpy']
LICENSE = "BSD-3-Clause"

AUTHOR = 'MBIRJAX development team'
EMAIL = "buzzard@purdue.edu"
PACKAGE_DIR = "MBIRJAX"

setup(install_requires=REQUIRES,
      zip_safe=False,
      name=NAME,
      version=VERSION,
      description=DESCR,
      author=AUTHOR,
      author_email=EMAIL,
      license=LICENSE,
      packages=find_packages(include=['mbirjax']),
      )

