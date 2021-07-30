"""
This code trains Sentence BERT model based on NLI dataset
"""

from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader
import math
import json
from pathlib import Path


def load_samples(jsonl_file, label_mapper):
    samples = []
    for line in open(jsonl_file):
        item = json.loads(line)
        sample = InputExample(texts=[item["premise"], item["hypothesis"]], label=label_mapper[item["label"]])
        samples.append(sample)
    return samples


def main(
    base_model, output_model, output_eval, train_data, valid_data, test_data,
    epochs=1, evaluation_steps=1000, batch_size=8
):
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
    train_loss = losses.SoftmaxLoss(
        model=model,
        sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
        num_labels=len(label_mapper)
    )
    evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(valid_samples)
    warmup_steps = math.ceil(len(train_dataloader) * 0.1)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=epochs,
        evaluation_steps=evaluation_steps,
        warmup_steps=warmup_steps,
        output_path=output_model,
    )

    # Test model
    test_model = SentenceTransformer(output_model)
    test_evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(
        test_samples,
        batch_size=batch_size
    )
    Path(output_eval).mkdir(exist_ok=True)
    test_evaluator(test_model, output_path=output_eval)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
