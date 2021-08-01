# Sentence BERT Japanese

This repository contains training script for Sentence BERT Japanese models.

## Prepare environment

```sh
$ docker container run --gpus all --ipc=host --rm -it -v $(pwd):/work -w /work nvidia/cuda:11.1-devel-ubuntu20.04 bash
(container)$ apt update && apt install -y python3 python3-pip git wget zip
(container)$ pip3 install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
(container)$ pip3 install -r requirements.txt
```

## Data

This model uses [Japanese SNLI data](https://nlp.ist.i.kyoto-u.ac.jp/index.php?%E6%97%A5%E6%9C%AC%E8%AA%9ESNLI%28JSNLI%29%E3%83%87%E3%83%BC%E3%82%BF%E3%82%BB%E3%83%83%E3%83%88) released under CC BY-SA 4.0 .

```sh
$ bash build_data.sh
```

Check the data sha1sum.

```sh
$ sha1sum data/JSNLI.zip
d6c9b45e8e6df03959f38cfbb58c31a747d6d12f  data/JSNLI.zip
```

{train,val,test}.jsonl data are prepared under a `data` directory.

## Train

```sh
$ python3 train.py --base_model colorfulscoop/bert-base-ja --output_model model --train_data data/train.jsonl --valid_data data/val.jsonl --test_data data/test.jsonl --epochs 1 --evaluation_steps=5000 --batch_size 8 --seed 1000 --use_amp
```

## Example usage

```sh
$ python semsearch.py --model model
======
Query: 走るのが趣味です
0.9029 外をランニングするのが好きです
0.7534 運動はそこそこです
0.5894 走るのは嫌いです
0.5451 天ぷらが食べたい
0.5335 りんごが食べたい
0.4970 海外旅行に行きたい
0.4268 揚げ物は食べたくない
======
Query: 外国を旅したい
0.9073 海外旅行に行きたい
0.7153 運動はそこそこです
0.6544 外をランニングするのが好きです
0.5313 天ぷらが食べたい
0.4653 りんごが食べたい
0.4413 揚げ物は食べたくない
0.4154 走るのは嫌いです
======
Query: 揚げ物が食べたい
0.9118 天ぷらが食べたい
0.7990 りんごが食べたい
0.6382 運動はそこそこです
0.5176 海外旅行に行きたい
0.5028 揚げ物は食べたくない
0.4898 外をランニングするのが好きです
0.4168 走るのは嫌いです
```

## Upload to Hugging Face Model Hub

Finally, upload the trained model to HuggingFace's model hub. Following the official document, the following process is executed.

First, create a repository named "sbert-base-ja" from HuggingFace's website.

Then, prepare git lfs. In a MacOS environment, git lfs can be installed as follows.

```sh
$ brew install git-lfs
$ git lfs install
Updated git hooks.
Git LFS initialized.
```

Then clone repository to local

```sh
$ git clone https://huggingface.co/colorfulscoop/sbert-base-ja
```

Copy model without evaluation result.

```sh
$ cp -r model/* sbert-base-ja
$ rm -r sbert-base-ja/eval
```

Copy model card and changelog files

```sh
$ cp model_card.md sbert-base-ja/README.md
$ cp CHANGELOG.md sbert-base-ja
```

Finally commit it and push to Model Hub.

```sh
$ cd sbert-base-ja
$ git add .
$ git commit -m "Add models and model card"
$ git push origin
```
