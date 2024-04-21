# Inferring Iterated Function Systems Approximately from Fractal Images (IJCAI 2024)
This is the official repository for the following paper:

>**Inferring Iterated Function Systems Approximately from Fractal Images (IJCAI 2024)**  [[paper]](https://openreview.net/pdf?id=wB2R7QQncw)<br>
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

Then use the following command to process the dataset. By doing so, we can train more quickly.
```
cd DataLoader
python Data_Lsystem.py
```

The processed data will be saved in ```Data_Generation/L_system/100_padding``` folder.

#### FractalDB dataset
You can download the FractalDB from [this](https://hirokatsukataoka16.github.io/Pretraining-without-Natural-Images/#dataset) and put the data under ```Data_Generation/FractalDB``` folder.


The split of data can be download from [this](https://drive.google.com/drive/folders/1zQ70SWF0BJS2aLpBE1-W41VUf7VPafW6?usp=drive_link). You can put those floders under your ```Data_Split``` folder.

### Train
Run the following command to train the model:
```
python train.py
```
You can change the Model you want to use, we use the DenseNet as defalut.

### Test
You can use your own training model, or can download our pre-train [model](https://drive.google.com/drive/folders/14OmoIQdZU_RhWo0uhW4rN2XBAn5I4vS9?usp=drive_link).
Run the following command can do the evaluation on the Julia dataset.
```
python eval_Julia.py
```

Run the following command can do the evaluation on the L-system dataset.
```
python eval_Lsystem.py
```

### Citation

### Aknowledgement


