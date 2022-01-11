from setuptools import setup
import os 
setup(
   name='SolidMechanicsTools',
   version='0.0.1',
   description='Solid Mechanics Toolbox for FEniCS',
   author='O.T. Turan',
   author_email='taylanozgurturan@gmail.com',
   packages=setuptools.find_packages(),
   include_package_data=True,
)

#setup(
#   name='SolidMechanicsTools',
#   version='0.0.1',
#   description='Solid Mechanics Toolbox for FEniCS',
#   author='O.T. Turan',
#   author_email='taylanozgurturan@gmail.com',
#   packages=['SolidMechanicsTools'],
#   package_data={
#       'SolidMechanicsTools': [os.path.join('materials','*.py'),\
#               os.path.join('src','*.py'),os.path.join('models','*.py')],
#       },
#
#)
