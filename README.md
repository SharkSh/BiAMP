# BiAMP
Source code for the paper "BiAMP: Bidirectional Generation of Antimicrobial Peptides and Fine-Grained Functional Texts Leveraging Diffusion and Language Models"

## Installation

Create a Python virtual environment and install the required packages:
```
pip install -r requirements.txt
```

## AMP-to-Text 
A deep learning model for converting antimicrobial peptide (AMP) sequences to text descriptions.
```
cd AMP2Text
```
### Project Structure
```
AMP-to-Text/
├── train.py                 # Training script
├── infer_batch.py           # Batch inference script
├── prot_gpt2_models/        # Model checkpoints
├── prediction_result/       # Inference results directory
└── data/                    # training data
```
### Training
To train the AMP-to-Text model
```
python ./train.py
```
### Inference
To run AMP-to-Text inference<br>
Save your antimicrobial peptide sequences to ./prediction_result/generated.csv
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

