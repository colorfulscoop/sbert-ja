"""
This code trains Sentence BERT model based on NLI dataset
"""

from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader
import math
import json
import os
import transformers
import logging


def load_samples(jsonl_file, label_mapper):
    samples = []
    for line in open(jsonl_file):
        item = json.loads(line)
        sample = InputExample(texts=[item["premise"], item["hypothesis"]], label=label_mapper[item["label"]])
        samples.append(sample)
    return samples


def main(
    base_model, output_model, train_data, valid_data, test_data,
    epochs=1, evaluation_steps=1000, batch_size=8, seed=None,
    use_amp=False,
):
    logging.basicConfig(level=logging.INFO)

    if seed:
        transformers.trainer_utils.set_seed(0)

    # Prepare model
    model = SentenceTransformer(base_model)

    # Prepare data
    label_mapper = {
        "contradiction": 0,
        "entailment": 1,
        "neutral": 2,
    }

    train_samples = load_samples(train_data, label_mapper)
    valid_samples = load_samples(valid_data, label_mapper)
    test_samples = load_samples(test_data, label_mapper)

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=batch_size)
    valid_dataloader = DataLoader(valid_samples, shuffle=False, batch_size=batch_size)
    test_dataloader = DataLoader(test_samples, shuffle=False, batch_size=batch_size)
    loss = losses.SoftmaxLoss(
        model=model,
        sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
        num_labels=len(label_mapper)
    )
    # See https://github.com/UKPLab/sentence-transformers/issues/27 about how to use LabelAccuracyEvaluator
    evaluator = evaluation.LabelAccuracyEvaluator(valid_dataloader, softmax_model=loss, name="val")
    warmup_steps = math.ceil(len(train_dataloader) * 0.1)

    model.fit(
        train_objectives=[(train_dataloader, loss)],
        evaluator=evaluator,
        epochs=epochs,
        evaluation_steps=evaluation_steps,
        warmup_steps=warmup_steps,
        output_path=output_model,
        use_amp=use_amp,
    )

    # Test model
    test_model = SentenceTransformer(output_model)
    test_model.to(model.device)
    loss.model = test_model
    test_evaluator = evaluation.LabelAccuracyEvaluator(test_dataloader, softmax_model=loss, name="test")
    test_evaluator(test_model, output_path=os.path.join(output_model, "eval"))


if __name__ == "__main__":
    import fire

    fire.Fire(main)
