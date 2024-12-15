# VLM_TriTraining

This repository implements a Tri-Training framework leveraging Vision-Language Models (VLMs) as base estimators. The framework is evaluated on the base-to-new benchmark to demonstrate its effectiveness.

## How to Install 

This code is built on top of [PromptSRC](https://github.com/muzairkhattak/PromptSRC) and the awesome toolbox [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch). So you need to install the `dassl` environment first. (**Note:** This repository already contains a modified version of [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch), so don't clone it separately.)

```bash
cd Dassl.pytorch/

# Create a conda environment
conda create -y -n dassl python=3.8

# Activate the environment
conda activate dassl

# Install torch (requires version >= 1.8.1) and torchvision
# Please refer to https://pytorch.org/ if you need a different cuda version
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

# Install dependencies
pip install -r requirements.txt

# Install this library (no need to re-build if the source code is modified)
python setup.py develop

cd ..
```

Then install the requirements of [PromptSRC](https://github.com/muzairkhattak/PromptSRC)

```bash
pip install -r requirements.txt
pip install setuptools==59.5.0
```

## How to Run

We provide the running script `VLM_TriTraining.sh`, which allow you to easily run the experiment. Make sure you change the path in Ensure you modify the file paths in the following scripts before running:

- `scripts/tritraining/base2novel_train.sh`
- `scripts/tritraining/base2novel_test_base.sh` 
- `scripts/tritraining/base2novel_test_new.sh`

and run the commands under the main directory `VLM_TriTraining`.

```bash
sh VLM_TriTraining.sh
```