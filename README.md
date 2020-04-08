# Introduction 

Multiple idcard detection

# Requirements

`pip install requirements.txt`

# Demo

`cd src/`

```
CUDA_VISIBLE_DEVICES=0 python demo.py
                            --model_path {MODEL_PATH}
                            --image_dir {IMAGE_DIR}
                            --output_dir {OUTPUT_DIR}
                            [--save_heatmap]
                            [--save_paf]
```

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
filename,width,height,tlx,tly,trx,try,brx,bry,blx,bly
a.jpg,624,1001,572,91,577,871,75,934,46,45
b.jpg,600,800,123,123,456,123,456,456,123,456
...
```

## 2. Training


