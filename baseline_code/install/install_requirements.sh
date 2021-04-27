#!/bin/bash
### install requirements for pstage3 baseline
# pip requirements
pip install datasets==1.5.0
pip install transformers==4.5.0
pip install tqdm==4.41.1
pip install pandas==1.1.4
pip install scikit-learn==0.24.1
pip install konlpy==0.5.2

# faiss install (if you want to)
conda install -c conda-forge faiss-gpu
