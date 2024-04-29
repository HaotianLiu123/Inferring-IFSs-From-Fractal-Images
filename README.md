# Inferring Iterated Function Systems Approximately from Fractal Images (IJCAI 2024)
This is the official repository for the following paper:

>**Inferring Iterated Function Systems Approximately from Fractal Images (IJCAI 2024)**
 <br>Haotian Liu, Dixin Luo, Hongteng Xu<br>
 Accepted by IJCAI 2024.
 
![Scheme](/assets/scheme.png "Learning Scheme")

## Model
In this work, we learn a multi-head auto-encoding model to infer typical IFSs approximately based on fractal images. The proposed model leverages two decoding heads to infer sequential and non-sequential parameters of different IFSs, and consider one more image decoding head to reconstruct input fractal images. We design a semi-supervised learning paradigm to learn the proposed model, making unlabeled
fractal images available during training. Our method provides a promising solution to infer Julia Set and L-system approximately from fractal images.

## Prepare

### Data
#### Julia Set
To generate Julia set, you just need to run the following command:
```
cd Data_Generation/Julia_Set/
python generate_data_randomly.py
```
The generated images will be saved in ```Data_Generation/Julia_Set/Data``` folder.

#### L-system based Data set
To generate L-system dataset, you just need to run the following command:
```
cd Data_Generation/L_system/
python draw.py
```
The generated images will be saved in ```Data_Generation/L_system/Data``` folder.

#### FractalDB dataset
You can download the FractalDB from [this](https://hirokatsukataoka16.github.io/Pretraining-without-Natural-Images/#dataset) and put the data under ```Data_Generation/FractalDB``` folder.

## Train

## Test

### Citation

### Aknowledgement


