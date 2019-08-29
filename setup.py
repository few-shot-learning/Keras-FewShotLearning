from setuptools import find_packages, setup

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='keras-fsl',
    version='0.0.1',
    description='Some State-of-the Art few shot learning algorithms Tensorflow 2.0',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Clément Walter',
    author_email='clementwalter@icloud.com',
    url='https://github.com/ClementWalter/Keras-FewShotLearning',
    license='MIT',
    install_requires=['pandas>=0.24.2'],
    extras_require={'publish': ['bumpversion>=0.5.3', 'twine>=1.13.0']},
    packages=find_packages(),
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Operating System :: OS Independent',
    ],
)
