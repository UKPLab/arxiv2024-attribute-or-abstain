#!/bin/bash

# Install dependencies
export VLLM_VERSION=0.4.0
export PYTHON_VERSION=311
pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu118-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux1_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu118
pip install hydra-core
pip install pytorch-lightning
pip install nltk
pip install rouge-score
pip install intertext-graph@git+https://github.com/UKPLab/intertext-graph.git@2516bd20a7153e825b89206dac0ebeb7d1ab7302
pip install git-python
pip install langchain
pip install peft
pip install rank-bm25
pip install sentence-transformers
pip install lm-format-enforcer
pip install tensorboard
pip install matplotlib
pip install bert-score
pip install rouge-score
pip install openai
module load cuda/11.8
pip install flash-attn --no-build-isolation

# Download NLTK punkt tokenizer
python -c "import nltk;nltk.download('punkt')"