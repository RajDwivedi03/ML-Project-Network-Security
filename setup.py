from setuptools import setup,find_packages
from typing import List

HYPEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:

    requirements=[]
    with open('requirements.txt') as file_obj:
      requirements=file_obj.readlines()
      requirements=[req.replace("/n","")for req in requirements]
    
      if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)


    return requirements    
setup(
    name='NETWORKSECURITY',
    version='1.0.0.0',
    packages=find_packages(),
    install_requires=get_requirements('requirement.txt'),
    author='Raj Dwivedi',
    author_email='rajdwivedi4814@gmail.com'
    )