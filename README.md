# Sentence BERT Japanese

$ pip install torch==1.8.1 sentence-transformers==2.0.0 fire==0.4.0

## Data

This model uses [Japanese SNLI data](https://nlp.ist.i.kyoto-u.ac.jp/index.php?%E6%97%A5%E6%9C%AC%E8%AA%9ESNLI%28JSNLI%29%E3%83%87%E3%83%BC%E3%82%BF%E3%82%BB%E3%83%83%E3%83%88) released under CC BY-SA 4.0 .

```sh
$ bash build_data.sh
```

Check the data sha1sum

```sh
$ sha1sum data/JSNLI.zip
d6c9b45e8e6df03959f38cfbb58c31a747d6d12f  data/JSNLI.zip
```

Then {train,val,test}.jsonl datasets are prepared under a `data` directory.

## Train

```sh
docker container run --gpus all --ipc=host --rm -it -v $(pwd):/work -w /work nvidia/cuda:11.1-devel-ubuntu20.04 bash
pip3 install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install -r requirements.txt
```

```sh
$ python train.py --base_model colorfulscoop/bert-base-ja --output_model model --output_eval eval --train_data data/train.jsonl --valid_data data/valid.jsonl --test_data data/test.jsonl --epochs 1 --evaluation_steps=1000 --batch_size 8
```