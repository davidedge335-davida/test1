from setuptools import find_packages, setup

packages = find_packages()
# ``models/backbone`` intentionally remains an implicit namespace package to
# match the upstream layout, so add it and its concrete E3CoverNet child
# explicitly; ``find_packages`` only discovers directories with __init__.py.
for namespace_package in (
        'e3covernet.models.backbone',
        'e3covernet.models.backbone.e3covernet'):
    if namespace_package not in packages:
        packages.append(namespace_package)

setup(name='E3CoverNet',
      version='2.0.0',
      description='Progressive E(3)-aware 3D vision-language alignment.',
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
                        'transformers',
                        'shapely',
                        'pyyaml'
                        ],
      packages=packages,
      python_requires='>=3.8',
      zip_safe=False)
