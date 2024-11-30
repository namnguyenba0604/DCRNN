## Tải thư viện trong Requirements
## Data Preparation
Tải `metr-la.h5` and `pems-bay.h5` từ [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) bỏ vào trong data folder

## Tạo 2 folder METR-LA và PEMS-BAY trong folder Data
```bash
# METR-LA
python -m scripts.generate_training_data --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5

# PEMS-BAY
python -m scripts.generate_training_data --output_dir=data/PEMS-BAY --traffic_df_filename=data/pems-bay.h5
```

## Graph Construction
```bash
python -m scripts.gen_adj_mx  --sensor_ids_filename=data/sensor_graph/graph_sensor_ids.txt --normalized_k=0.1\
    --output_pkl_filename=data/sensor_graph/adj_mx.pkl
```
Besides, the locations of sensors in Los Angeles, i.e., METR-LA, are available at [data/sensor_graph/graph_sensor_locations.csv](https://github.com/liyaguang/DCRNN/blob/master/data/sensor_graph/graph_sensor_locations.csv), and the locations of sensors in PEMS-BAY are available at [data/sensor_graph/graph_sensor_locations_bay.csv](https://github.com/liyaguang/DCRNN/blob/master/data/sensor_graph/graph_sensor_locations_bay.csv).

## Run the Pre-trained Model on METR-LA

```bash
# METR-LA
python run_demo.py --config_filename=data/model/pretrained/METR-LA/config.yaml

# PEMS-BAY
python run_demo.py --config_filename=data/model/pretrained/PEMS-BAY/config.yaml
```
The generated prediction of DCRNN is in `data/results/dcrnn_predictions`.


## Model Training

Here are commands for training the model on `METR-LA` and `PEMS-BAY` respectively. 

```bash
# METR-LA
python dcrnn_train.py --config_filename=data/model/dcrnn_la.yaml

# PEMS-BAY
python dcrnn_train.py --config_filename=data/model/dcrnn_bay.yaml
```


## Eval baseline methods
```bash
# METR-LA
python -m scripts.eval_baseline_methods --traffic_reading_filename=data/metr-la.h5
```
