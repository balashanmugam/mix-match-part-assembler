# mix-match-part-assembler
Course project for CMPT 464/764

## Running the mixer

Change the input and output directories at the top of the file then run it with

python chair-mixer.py
To create images change the settings at the top of the file then run

python create-images.py

## How to run scorer part of the code

### MVCNN/train.py
Running it will create the VGG model and train it for 6 epochs and then save weights in checkpoints.

### MVCNN/evaluate_sample.py
Running it will load in the weights and evaluate images in a given folder then it will output a numpy array of all the predicted scores.
