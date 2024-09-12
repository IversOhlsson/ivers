from setuptools import setup, find_packages

setup(
    name='ivers',
    version='0.1.13',
    packages=find_packages(),
    description='Python package to stratify split datasets based on endpoint distributions, also 2 different temporal splits. Chemprop compatible.',
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
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    keywords='chemprop chemistry data science dataset splitting stratification temporal splits ivers',
    project_urls={
        'Documentation': 'http://github.com/iversohlsson/ivers/docs/_build/html/index.html',
        'Source': 'http://github.com/iversohlsson/ivers',
    },
)
from setuptools import setup, find_packages

