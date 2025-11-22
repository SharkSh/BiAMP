# BiAMP
Source code for the paper "BiAMP: Bidirectional Generation of Antimicrobial Peptides and Fine-Grained Functional Texts Leveraging Diffusion and Language Models"

## Installation

Create a Python virtual environment and install the required packages:

```
pip install -r requirements.txt
```

## AMP-to-Text 
A deep learning model for converting antimicrobial peptide (AMP) sequences to text descriptions.
Project Structure
```
cd AMP2Text
```
to train AMP-to-text
```
python ./train.py
```
to run AMP-to-Text
```
python ./infer_batch.py
```

## Run Text-to-AMP 

```
cd Text-to-AMP
torchrun --nproc-per-node=1 ./generation/scripts/infill.py
```

## Decode generated embeddings to AMPs 
```
python  python ./generation/scripts/decode_to_seq.py
```

