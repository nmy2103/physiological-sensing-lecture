from setuptools import setup, find_packages

setup(
    name = 'heartview',
    version = '2.0.1',
    packages = find_packages(),
    install_requires = [
        'numpy==1.26.4',
        'pandas==2.2.2',
        'scipy==1.14.0',
        'tqdm==4.65.0',
    ]
)