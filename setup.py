from setuptools import find_packages, setup

LONG_DESCRIPTION = (
    'Desc.'
)


setup(
        name='mhealth',
        version='0.0.3',
        packages=find_packages(where='src'),
        package_dir={'': 'src'},
        url='https://github.com/callumstew/pymhealth',
        author='Callum Stewart',
        author_email='callum.stewart@kcl.ac.uk',
        description='An mHealth processing and feature extraction library',
        long_description=LONG_DESCRIPTION,
        install_requires=[
            'numpy',
            'numba',
            'scipy',
            'hdbscan'
        ],
        classifiers=[
            'Programming Language :: Python',
        ]
)
