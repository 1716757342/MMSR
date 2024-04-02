# MMSR: Symbolic Regression is a multi-modal information fusion task

MMSR is a new symbolic regression algorithm. It regards symbolic regression as a task of multi-modal information fusion. Moreover, the experimental results show that it achieves the effect of SOTA on multiple symbolic regression datasets.

[//]: # ([Paper]&#40;https://arxiv.org/pdf/2205.15764&#41;&nbsp;&nbsp;&nbsp;)

[//]: # ([Web]&#40;https://vastlik.github.io/symformer/&#41;&nbsp;&nbsp;&nbsp;)

[//]: # ([Demo]&#40;https://colab.research.google.com/github/vastlik/symformer/blob/main/notebooks/symformer-playground.ipynb&#41;)

<br>

[//]: # (链接: https://pan.baidu.com/s/1VzBbL48u0F7mt20mL-FPNA?pwd=p2ir 提取码: p2ir )

[//]: # (--来自百度网盘超级会员v7的分享)
[//]: # ([![Open In Colab]&#40;https://colab.research.google.com/assets/colab-badge.svg?style=for-the-badge&#41;]&#40;https://pan.baidu.com/s/1VzBbL48u0F7mt20mL-FPNA?pwd=p2ir&#41;)
![Python Versions](https://img.shields.io/badge/Python-3.7%20%7C%203.8%20%7C%203.9-blue)
## Model download

You can download the pre-trained model from the following link. https://pan.baidu.com/s/1VzBbL48u0F7mt20mL-FPNA?pwd=p2ir
<br>

## Getting started

Begin by setting up a virtual environment with Python 3.9. Once it's active, proceed by executing the following command at the root level of the repository:



```
pip install -r requirements.txt
```

## Data acquisition

To produce a one-dimensional dataset for training the univariate model, execute the commands below:
```
python -m MMSR generate-dataset \
    --output-dir general/train \
    --dataset-size 130000000 \
    --n-processes 128 \
    --seed 1234
python -m MMSR generate-dataset \
    --output-dir general/valid \
    --dataset-size 10000 \
    --n-processes 128 \
    --seed 5678
```

To generate a two-dimensional dataset (used to train the bivariate model) run the following commands:

```
python -m MMSR generate-dataset \
    --output-dir general/train \
    --dataset-size 100000000 \
    --n-processes 128 \
    --seed 1234 \
    --num-variables 2
python -m MMSR generate-dataset \
    --output-dir general/valid \
    --dataset-size 10000 \
    --n-processes 128 \
    --seed 5678 \
    --num-variables 2
```

For further hyperparameters see `python -m MMSR generate-dataset --help`.

## Running the inference

You can specify your own path to trained models using --model, including 'MMSR-univariate' or 'MMSR-bivariate'.
### Single equation

To run a single equation:

```
python -m MMSR predict --model /path/to/MMSR-model 'x**2 + x'
```

Output:

```
Function: x^2 + x
R2: 1.0
Relative error: 4.21475839274614e-17
```

Additionally, you have the option to switch to a model of your choice.

### Benchmark dataset

To run the benchmark use command bellow:

```
python -m MMSR evaluate-benchmark --univariate-model /path/to/MMSR-univariate --bivariate-model /path/to/MMSR-bivariate
```

### Evaluation on dataset

To run the evaluation on dataset run the following:

```
python -m MMSR evaluate --model /path/to/MMSR-univariate --test-dataset-path /path/to/datast
```


## Training a model from scratch

To train a model run the following:

```
python -m MMSR train \
    --config configs/{config name}.json \
    --dataset-path /path/to/train/dataset/ \
    --dataset-valid-path /path/to/valid/dataset/
```

where `{config name}` is one of the files contained in the `configs` directory.

## Citation

