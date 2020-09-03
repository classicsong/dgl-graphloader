#!/usr/bin/env python3

# -*- coding: utf-8 -*-
#
# setup.py
#
# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from setuptools import find_packages
from setuptools import setup

VERSION = '0.0.1'
setup(
    name='dgl-graphloader',
    version=VERSION,
    description='DGL Graph Loader',
    maintainer='DGL Team',
    maintainer_email='classicxsong@gmail.com',
    packages=find_packages(),
    install_requires=[
        'scikit-learn>=0.23.2',
        'scapy>=2.4.3',
        'dgl>=0.5.0'
    ],
    url='https://github.com/classicsong/dgl-graphloader.git',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
    ],
    license='APACHE'
)
