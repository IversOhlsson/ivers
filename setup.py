from setuptools import setup, find_packages

setup(
    name='ivers',
    version='0.3.0',
    packages=find_packages(),
    description='Python package to stratify split datasets based on endpoint distributions',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Philip Ivers Ohlsson',
    author_email='philip.iversohlsson@gmail.com',
    url='http://github.com/iversohlsson/ivers',
    install_requires=[
        'pandas',
        'scikit-learn'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
from setuptools import setup, find_packages

