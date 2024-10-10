from setuptools import setup, find_packages
import os

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

extra_files = package_files('whisperx_transcriber/models')

setup(
    name="whisperx_transcriber",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={'': extra_files},
    install_requires=[
        "whisperx",
        "pyannote.audio",
        "PyQt5",
        "torch",
        "pyyaml",
    ],
    entry_points={
        'console_scripts': [
            'whisperx_transcriber=whisperx_transcriber.gui:main',
        ],
    },
)