from setuptools import find_packages,setup
from typing import List

def get_req(file_path):
    req=[]
    with open(file_path) as file_obj:
        req=file_obj.readlines()
        req=[i.replace('\n','') for i in req]

        if '-e .' in req:
            req.remove('-e .')


setup(
    packages=find_packages(),
    install_requires=get_req('requirements.txt')
    )