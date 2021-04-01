from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

def parse_requirements(fn):
    with open(fn) as f:
        return [req for req in f.read().strip().split('\n') if "==" in req and "#" not in req]


long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='bertnlp',
    version='0.0.11',
    description="BERT toolkit is a Python package that performs various NLP tasks using Bidirectional Encoder Representations from Transformers (BERT) related models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Shuanglu Dai",
    author_email='shud@microsoft.com',
    packages=find_packages(),
    include_package_data=True,
    install_requires=parse_requirements('requirements.txt'),
    zip_safe=False,
    keywords='bertnlp',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ]
)

