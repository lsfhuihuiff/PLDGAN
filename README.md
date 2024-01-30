# PLDGAN
The code for “PLDGAN: Portrait Line Drawing Generation with Prior Knowledge and Conditioning Target“

## Installation

This code requires PyTorch 1.0 and python 3+. Please install dependencies by
```bash
pip install -r requirements.txt
```

## Generating Images Using Pretrained Model

Once the dataset is ready, the result images can be generated using pretrained models.

1. Download the tar of the pretrained models from the [Google Drive Folder](https://drive.google.com/file/d/1_MwN9Q32ka1STS3zDDslhZxPjwUcLP_j/view?usp=share_link), save it in 'checkpoints/', and run

    ```
    cd checkpoints
    unzip sketch.zip
    cd ../
    ```

2. Generate images using the pretrained model.
    ```bash
    python test.py --no_instance --no_flip
    ```
