# Related Memory Network (RMN)

<img src = "./figure/RMN.png" width="850">

## Prerequisites
* Python 3.6
* Tensorflow 1.3.0
* dependencies
  * `pip install tqdm colorlog`

## Usage

### 1. prepare data

To process bAbI story-based QA dataset, run:
```
$ python preprocessor.py --data story
```

To process bAbI dialog dataset, run:
```
$ python preprocessor.py --data dialog
```

### 2. train model

To train RMN on bAbI story-based QA dataset, run:
```
$ python ./babi_story/train.py
```

To train RMN on bAbI dialog dataset task 4, run:
```
$ python ./babi_dialog/train.py --task 4
```
To use match, use_match flag is required:
```
$ python ./babi_dialog/train.py --task 4 --use_match True
```
To test on OOV dataset, is_oov flag is required:
```
$ python ./babi_dialog/train.py --task 4 --is_oov True
```
