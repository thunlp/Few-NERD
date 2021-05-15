

# Few-NERD: Now Only a Few-shot NER Dataset

This is the source code of the ACL-IJCNLP 2021 paper:  **Few-NERD: A Few-shot Named Entity Recognition Dataset**.  

Check out the (website)[https://ningding97.github.io/fewnerd/] of Few-NERD. The code implements 3 models (ProtoBERT, NNShot, StructShot).

## Overview

Few-NERD is a large-scale, fine-grained manually annotated named entity recognition dataset, which contains 8 coarse-grained types, 66 fine-grained types, 188,200 sentences, 491,711 entities and 4,601,223 tokens. Three benchmark tasks are built, one is supervised (Few-NERD (SUP)) and the other two are few-shot (Few-NERD (INTRA) and Few-NERD (INTER)). 

The schema of Few-NERD is

 ![few-nerd](/Users/dingning/Desktop/few-nerd.png)

## Requirements

 Run the following script to install the remaining dependencies,

`pip install -r requirements.txt`

## Data 

To obtain the three benchmarks datasets of Few-NERD, simply run the bash file `data/download.sh`

`		bash download.sh`

The data are pre-processed into the typical NER data forms as below (`token\tlabel`). Each dataset should contain train.txt, val.txt, test.txt 3 separate files.

```tex
Between	O
1789	O
and	O
1793	O
he	O
sat	O
on	O
a	O
committee	O
reviewing	O
the	O
administrative	other-law
constitution	other-law
of	other-law
Galicia	other-law
to	O
little	O
effect	O
.	O
```

## Structure

The structure of our project is:

```
--util
| -- framework.py
| -- data_loader.py
| -- viterbi.py             # viterbi decoder for structshot only
| -- word_encoder
| -- fewshotsampler.py

-- proto.py                 # prototypical model
-- nnshot.py                # nnshot model

-- train_demo.py            # main training script
```



## Key Implementations

#### Sampler

As established in our paper, we design an *N way K~2K shot* sampling strategy in our work , the  implementation is sat `util/fewshotsampler.py`.

#### ProtoBERT

 Prototypical nets with BERT is implemented in `model/proto.py`

 



## How to Run

- The conducted experiments cover 3 models across 4 different few-shot settings. 
- The parameter `--model` is used to specify which model to run. The 3 options are `proto`, `nnshot` and `structshot`.
- The parameters `--trainN` and `--N` are used to specify the num of ways in few-shot, in support and query set respectively. `--K` and `--Q` are for num of shots in support and query set, respectively.
- For hyperparameter `--tau` in structshot, we use `0.32` in 1-shot setting, `0.318` for 5-way-5-shot setting, and `0.434` for 10-way-5-shot setting.
- Take structshot for example, the expriments can be run as follows.

**5-way-1~5-shot**

```
python3 train_demo.py  --train data/mydata/train-inter.txt \
--val data/mydata/val-inter.txt --test data/mydata/test-inter.txt \
--lr 1e-3 --batch_size 2 --trainN 5 --N 5 --K 1 --Q 1 \
--train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length 60 --model structshot --tau 0.32
```

**5-way-5~10-shot**

```
python3 train_demo.py  --train data/mydata/train-inter.txt \
--val data/mydata/val-inter.txt --test data/mydata/test-inter.txt \
--lr 1e-3 --batch_size 2 --trainN 5 --N 5 --K 5 --Q 5 \
--train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length 60 --model structshot --tau 0.318
```

**10-way-1~5-shot**

```
python3 train_demo.py  --train data/mydata/train-inter.txt \
--val data/mydata/val-inter.txt --test data/mydata/test-inter.txt \
--lr 1e-3 --batch_size 2 --trainN 10 --N 10 --K 1 --Q 1 \
--train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length 60 --model structshot --tau 0.32
```

**10-way-5~10-shot**

```
python3 train_demo.py  --train data/mydata/train-inter.txt \
--val data/mydata/val-inter.txt --test data/mydata/test-inter.txt \
--lr 1e-3 --batch_size 2 --trainN 5 --N 5 --K 5 --Q 1 \
--train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length 60 --model structshot --tau 0.434
```



## Citation

If you use Few-NERD in your work, please cite our paper:

```
@inproceedings{ding2021few,
  title={Few-NERD:A Few-shot Named Entity Recognition Dataset},
  author={Ding, Ning and Xu, Guangwei and Chen, Yulin, and Wang, Xiaobin and Han, Xu and Xie, Pengjun and Zheng, Hai-Tao and Liu, Zhiyuan},
  booktitle={ACL-IJCNLP},
  year={2021}
}
```



