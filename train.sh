
CUDA_VISIBLE_DEVICES='1,2,4' python -m MMSR train \
    --config configs/bivariate.json \
    --dataset-path path/to/train/data/ \
    --dataset-valid-path path/to/valed/data/



