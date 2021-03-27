#! -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='tf2bert',
    version='1.0.0',
    description='a simple bert for beginner',
    long_description='tf2bert:https://github.com/boss2020/tf2bert',
    url='https://github.com/boss2020/tf2bert',
    author='xiaoguzai',
    author_email='474551240@qq.com',
    install_requires=['tensorflow>=2.0'],
    packages=find_packages()
)
