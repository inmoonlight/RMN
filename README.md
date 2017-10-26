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
