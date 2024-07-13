# Inferring Iterated Function Systems Approximately from Fractal Images (IJCAI 2024)
This is the official repository for the following paper:

>**Inferring Iterated Function Systems Approximately from Fractal Images (IJCAI 2024)**  <br>
 <br>Haotian Liu, Dixin Luo, Hongteng Xu<br>
 Accepted by IJCAI 2024.
 
![Scheme](/assets/scheme.png "Learning Scheme")

## Install

```commandline
pip install -r requirements.txt
```

## Model
In this work, we learn a multi-head auto-encoding model to infer typical IFSs approximately based on fractal images. The proposed model leverages two decoding heads to infer sequential and non-sequential parameters of different IFSs, and consider one more image decoding head to reconstruct input fractal images. We design a semi-supervised learning paradigm to learn the proposed model, making unlabeled
fractal images available during training. Our method provides a promising solution to infer Julia Set and L-system approximately from fractal images.

## Prepare

### Data
#### Julia Set
Run the following command to generate Julia set:
```
cd Data_Generation/Julia_Set/
python generate_data_randomly.py
```
The generated images will be saved in ```Data_Generation/Julia_Set/Data``` folder.

#### L-system based Data set
Run the following command to generate L-system dataset:
```
cd Data_Generation/L_system/
python draw.py
```
The generated images will be saved in ```Data_Generation/L_system/Data``` folder.

Then use the following command to process the dataset. By doing this, we can further process the data, thereby saving time on loading data during training.
```
cd Data_Generation/L_system/
python create_input_files.py
```

The processed data will be saved in ```Data_Generation/L_system/100_padding``` folder.

#### FractalDB dataset
You can download the FractalDB from [this](https://hirokatsukataoka16.github.io/Pretraining-without-Natural-Images/#dataset) and put the data under ```Data_Generation/FractalDB``` folder.


The split of data can be download from [this](https://drive.google.com/drive/folders/131g284E5_lqKDGp7XAFbCu82NFNoISNJ?usp=sharing). You should place the corresponding files in the folders that contain the respective images.


## Train
Run the following command to train the model. You can adjust the parameters inside according to your own needs.

[//]: # (```)

[//]: # (python train.py --data_folder Data_Generation/L_system/100_padding/ --Julia_data_folder Data_Generation/Julia_Set/Data/ --fractalDB_data_folder Data_Generation/FractalDB/)

[//]: # (```)

```
cd Methods
python train.py --data_folder  path/to/L_system/data --Julia_data_folder path/to/Julia_Set/data/ --fractalDB_data_folder path/to/FractalDB/data --save_dir path/to/save/checkpoint
```

## Evalation
Run the following command can do the evaluation on the Julia dataset and L-system dataset. You can adjust the parameters inside according to your own needs.
```
cd Methods
python eval.py --data_folder  path/to/L_system/data --Julia_data_folder path/to/Julia_Set/data/ --checkpoint path/to/save/checkpoint
```


## Citation

## Aknowledgement


