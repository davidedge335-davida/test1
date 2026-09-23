from setuptools import find_packages, setup

setup(name='E3CoverNet',
      version='0.1',
      description='Comprehension of localizing 3D objects in scenes.',
      url='https://github.com/E3CoverNet/E3CoverNet',
      author='E3CoverNet team',
      author_email='optas@cs.stanford.edu',
      license='MIT',
      install_requires=['scikit-learn',
                        'matplotlib',
                        'six',
                        'tqdm',
                        'pandas',
                        'plyfile',
                        'requests',
                        'symspellpy',
                        'termcolor',
                        'tensorboardX',
                        'shapely',
                        'pyyaml'
                        ],
      packages=find_packages(),
      zip_safe=False)
