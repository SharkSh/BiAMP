# BiAMP
Source code for the paper "BiAMP: Bidirectional Generation of Antimicrobial Peptides and Fine-Grained Functional Texts Leveraging Diffusion and Language Models"

## Installation

Create a Python virtual environment and install the required packages:

```
pip install -r requirements.txt
```

## Run AMP-to-Text 

```
python ./AMP2Text/infer_batch.py
```

## Run Text-to-AMP 

```
cd Text-to-AMP
torchrun --nproc-per-node=1 ./generation/scripts/infill.py
```


