#!/usr/bin/env python
from io import open
from setuptools import setup, find_packages

setup(
    name='tf2bert',
    version='1.0.0',
    description='a package for your bert using',
    long_description='使用keras实现你的bert项目',
    author='xiaoguzai',
    author_email='474551240@qq.com',
    license='apache2.0',
    url='http://github.com/liangdongchang/django-xadmin',
    download_url='http://github.com/liangdongchang/django-xadmin/master.zip',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'setuptools',
        'tensorflow'>=2.4.0
    ]
)