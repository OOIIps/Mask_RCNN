from setuptools import setup, find_packages

dependencies = [

"absl-py",
"astor",
"bleach",
"cycler",
"decorator",
"gast",
"grpcio",
"h5py",
"html5lib",
"Keras==2.1.5",
"kiwisolver",
"Markdown",
"matplotlib",
"networkx",
"numpy",
"opencv-contrib-python==3.4.0.12",
"Pillow",
"protobuf",
"pyparsing",
"python-dateutil",
"pytz",
"PyWavelets",
"PyYAML",
"scikit-image",
"scipy",
"six",
"tensorboard==1.7.0",
"tensorflow==1.7.0",
"termcolor",
"Werkzeug",
"imgaug"

]

packages = [
    package for package in find_packages() if package.startswith('mask_rcnn')
]

setup(name='mask_rcnn',
      version='0.1',
      description='Mask RCNN library',
      author='hanskrupakar',
      author_email='hansk@nyu.edu',
      license='Open-Source',
      url='https://www.github.com/hanskrupakar/Mask_RCNN',
      packages=packages,
      install_requires=dependencies,
      test_suite='test'
)
