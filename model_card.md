---
language: ja
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- feature-extraction
- sentence-similarity
widget:
    source_sentence: "走るのが趣味です"
    sentences:
        - 外をランニングするのが好きです
        - 運動はそこそこです
        - 走るのは嫌いです
license: cc-by-sa-4.0
---

# Sentence BERT base Japanese model

This repository contains a Sentence BERT base model for Japanese.

## Pretrained model

This model utilizes a Japanese BERT model [colorfulscoop/bert-base-ja](https://huggingface.co/colorfulscoop/bert-base-ja) v1.0 released under [Creative Commons Attribution-ShareAlike 3.0](https://creativecommons.org/licenses/by-sa/3.0/) as a pretrained model.

## Training data

[Japanese SNLI dataset](https://nlp.ist.i.kyoto-u.ac.jp/index.php?%E6%97%A5%E6%9C%AC%E8%AA%9ESNLI%28JSNLI%29%E3%83%87%E3%83%BC%E3%82%BF%E3%82%BB%E3%83%83%E3%83%88) released under [Creative Commons Attribution-ShareAlike 4.0](https://creativecommons.org/licenses/by-sa/4.0/) is used for training.

Original training dataset is splitted into train/valid dataset. Finally, follwoing data is prepared.

* Train data: 523,005 samples
* Valid data: 10,000 samples
* Test data: 3,916 samples

## Model description

This model utilizes `SentenceTransformer` model from the [sentence-transformers](https://github.com/UKPLab/sentence-transformers) .
The model detail is as below.

```py
>>> from sentence_transformers import SentenceTransformer
>>> SentenceTransformer("colorfulscoop/sbert-base-ja")
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False}) with Transformer model: BertModel
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False})
)
```

## Training

This model finetuned [colorfulscoop/bert-base-ja](https://huggingface.co/colorfulscoop/bert-base-ja) with Softmax classifier of 3 labels of SNLI. AdamW optimizer with learning rate of 2e-05 linearly warmed-up in 10% of train data was used. The model was trained in 1 epoch with batch size 8.

Note: in a original paper of [Sentence BERT](https://arxiv.org/abs/1908.10084), a batch size of the model trained on SNLI and Multi-Genle NLI was 16. In this model, the dataset is around half smaller than the origial one, therefore the batch size was set to half of the original batch size of 16.

Trainind was conducted on Ubuntu 18.04.5 LTS with one RTX 2080 Ti.

After training, test set accuracy reached to 0.8529.

Training code is available in [a GitHub repository](https://github.com/colorfulscoop/sbert-ja).

## Usage

First, install dependecies.

```sh
$ pip install sentence-transformers==2.0.0
```

Then initialize `SentenceTransformer` model and use `encode` method to convert to vectors.

```py
>>> from sentence_transformers import SentenceTransformer
>>> model = SentenceTransformer("colorfulscoop/sbert-base-ja")
>>> sentences = ["外をランニングするのが好きです", "海外旅行に行くのが趣味です"]
>>> model.encode(sentences)
```

## License

Copyright (c) 2021 Colorful Scoop

All the models included in this repository are licensed under [Creative Commons Attribution-ShareAlike 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

**Disclaimer:** Use of this model is at your sole risk. Colorful Scoop makes no warranty or guarantee of any outputs from the model. Colorful Scoop is not liable for any trouble, loss, or damage arising from the model output.

---

This model utilizes the folllowing pretrained model.

* **Name:** bert-base-ja
* **Credit:** (c) 2021 Colorful Scoop
* **License:** [Creative Commons Attribution-ShareAlike 3.0](https://creativecommons.org/licenses/by-sa/3.0/)
* **Disclaimer:** The model potentially has possibility that it generates similar texts in the training data, texts not to be true, or biased texts. Use of the model is at your sole risk. Colorful Scoop makes no warranty or guarantee of any outputs from the model. Colorful Scoop is not liable for any trouble, loss, or damage arising from the model output.
* **Link:** https://huggingface.co/colorfulscoop/bert-base-ja

---

This model utilizes the following data for fine-tuning.

* **Name:** 日本語SNLI(JSNLI)データセット
* **Credit:** [https://nlp.ist.i.kyoto-u.ac.jp/index.php?日本語SNLI(JSNLI)データセット](https://nlp.ist.i.kyoto-u.ac.jp/index.php?%E6%97%A5%E6%9C%AC%E8%AA%9ESNLI%28JSNLI%29%E3%83%87%E3%83%BC%E3%82%BF%E3%82%BB%E3%83%83%E3%83%88)
* **License:** [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
* **Link:** [https://nlp.ist.i.kyoto-u.ac.jp/index.php?日本語SNLI(JSNLI)データセット](https://nlp.ist.i.kyoto-u.ac.jp/index.php?%E6%97%A5%E6%9C%AC%E8%AA%9ESNLI%28JSNLI%29%E3%83%87%E3%83%BC%E3%82%BF%E3%82%BB%E3%83%83%E3%83%88)