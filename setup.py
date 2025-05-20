from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'AdaptiveSplitAnalysis'
LONG_DESCRIPTION = 'Repo storing the analyses and manuscript for the adaptivesplit approach'

# Setting up
setup(
    name="adaptivesplit",
    version="0.0.1",
    author="PNI Lab (Predictive NeuroImaging Laboratory)",
    author_email="<giuseppe.gallitto@uk-essen.de>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=["joblib==1.2.0", "matplotlib==3.6.0", "numpy==1.24.2", "pandas==1.5.1", "pygam==0.9.0",
                      "regressors==0.0.3", "scikit_learn==1.2.1", "scipy==1.11.3", "tqdm==4.64.1", "setuptools==80.7.1],
    keywords=['python', 'machine learning', 'neuroimaging', 'data splitting'],
    classifiers=[]
)
