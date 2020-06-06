# Spatial-Temporal Sequence to Sequence Model for Traffic Forecasting

A PyTorch implementation of *Spatial-Temporal Sequence to Sequence* model in the paper:  Forecast Network-Wide Traffic States for Multiple Steps Ahead: A Deep Learning Approach Considering Dynamic Non-Local Spatial Correlation and Non-Stationary Temporal Dependency (https://arxiv.org/abs/2004.02391).

## Requirements
- pytorch >= 1.2.0
- tensorboard >= 1.14.0
- scikit-learn
- statsmodels
- tqdm

## Data Preparation

### Speed Data 

 Step1: Download METR-LA and PEMS-BAY data from [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g) links provided by [DCRNN](https://github.com/liyaguang/DCRNN).

 Step2: Follow [DCRNN](https://github.com/liyaguang/DCRNN)'s scripts to preprocess data.

Run the following commands to generate train/test/val dataset at  `data/{METR-LA,PEMS-BAY}/{train,val,test}.npz`.
```bash
# Create data directories
mkdir -p data/{METR-LA,PEMS-BAY}

# METR-LA
python -m scripts.generate_training_data --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5

# PEMS-BAY
python -m scripts.generate_training_data --output_dir=data/PEMS-BAY --traffic_df_filename=data/pems-bay.h5
```

### Graph Data

The construction of  graph adjacency matrix is based on pre-calculated road network distances between sensors. Here we simply use the adjacency matrices provided by [DCRNN](https://github.com/liyaguang/DCRNN). These matrices are available in [/data](https://github.com/xlwang233/STSeq2Seq/data) folder.

## Model Training and Testing

This repo follows the [PyTorch Template](https://github.com/victoresque/pytorch-template), which uses `.json` file for parameter configuration. 

### Training

Run following command for model training.

```bash
# train STSeq2Seq 
python train.py -c config.json
```
Each epoch takes about 100 seconds for METR-LA under computing environment with one Core i7-7700K CPU and single NVIDIA RTX 2080Ti GPU. The training log and models will be saved in `saved/METR-LA_STSeq2Seq/` 

### Testing

Run following command to evaluate your trained model.

```bash
# test STSeq2Seq 
python test.py -r saved/METR-LA_STSeq2Seq/models/{time stamp}/model_best.pth
```

A pre-trained model for METR-LA is provided and can be run by:

```bash
# run pre-trained STSeq2Seq for METR-LA
python test.py -r pretrained/METR-LA/metr-la.pth
```

## Evaluate Baseline Methods

For neural network models, i.e. FNN and GRU, the training and testing procedure is similar to that of STSeq2Seq.

```bash
# train FNN/GRU
python train.py -c config_{FNN,GRU}.json

# test FNN/GRU
python test.py -r saved/METR-LA_{FNN,GRU}/models/{time stamp}/model_best.pth
```

For HA, ARIMA and SVR, go to `scripts/` directory and run

```bash
# METR-LA
python eval_baselines.py
```

Note that ARIMA and SVR are fitted independently on each sensor, thus would probably lead to intolerable computation time (on single machine).  Some workarounds may include: 

- consider using parallel and distributed computing tools (e.g. Apache Spark, which I have not tested its feasibility, though) or 
- use simpler models (e.g. use LinearSVR instead of SVR). 



 