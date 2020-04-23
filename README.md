# Introduction 

Multiple idcard detection

# Requirements

`pip install requirements.txt`

# Demo

`cd src/`

```
CUDA_VISIBLE_DEVICES=0 python demo_classes.py
                            --model_path {MODEL_PATH}
                            --image_dir {IMAGE_DIR}
                            --output_dir {OUTPUT_DIR}
                            [--save_heatmap]
                            [--save_paf]
```

eg: `CUDA_VISIBLE_DEVICES=0 python demo_classes.py --model_path 1000.ckpt --image_dir abc/ --output_dir def/`

# Quickstart

## 1. Dataset

Structure data root should be:

```
{DATA_ROOT}
|-- data
    |-- train
        |-- images
        label.csv
    |-- val
        |-- images
        |-- label.csv
```

### label.csv
```
filename,width,height,class,tlx,tly,trx,try,brx,bry,blx,bly
a.jpg,624,1001,0,572,91,577,871,75,934,46,45
b.jpg,600,800,123,1,123,456,123,456,456,123,456
...
```

## 2. Training

### 1: Modify config.py

`BATCH_SIZE`: batch_size when train and evaluation

`PRETRAINED`: use pretrained ckpt or not

`PRETRAINED_PATH`: path to pretrained ckpt

`MODEL_DIR`: folder save ckpt while training

`NUM_CLASSES`: number of object classes (note: each classes must have same joint)

`LABEL_MAP`: label map of object, must same as `class` in file `label.csv`

### 2: Training

`CUDA_VISIBLE_DEVICES=0 python main.py`



