from setuptools import find_packages, setup

LONG_DESCRIPTION = (
    'Desc.'
)


setup(
        name='mhealth',
        version='0.0.1',
        packages=find_packages(where='src'),
        package_dir={'mhealth': 'src/mhealth'},
        url='https://github.com/callumstew/pymhealth',
        author='Callum Stewart',
        author_email='callum.stewart@kcl.ac.uk',
        description='An mHealth processing and feature extraction library',
        long_description=LONG_DESCRIPTION,
        install_requires=[
            'numpy',
            'numba'
        ],
        classifiers=[
            'Programming Language :: Python',
        ]
)
