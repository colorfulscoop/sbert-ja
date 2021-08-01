from sentence_transformers import SentenceTransformer, util
import torch


def main(model, top_k=None):
    samples = [
        "外をランニングするのが好きです",
        "走るのは嫌いです",
        "運動はそこそこです",
        "海外旅行に行きたい",
        "天ぷらが食べたい",
        "りんごが食べたい",
        "揚げ物は食べたくない",
    ]
    model = SentenceTransformer(model)
    samples_embedding = model.encode(samples, convert_to_tensor=True)

    # Query sentences:
    queries = [
        "走るのが趣味です",
        "外国を旅したい",
        "揚げ物が食べたい",
    ]

    if top_k:
        top_k = min(top_k, len(samples))
    else:
        top_k = len(samples)
    for query in queries:
        query_embedding = model.encode(query, convert_to_tensor=True)

        scores = util.pytorch_cos_sim(query_embedding, samples_embedding)[0]
        result = torch.topk(scores, k=top_k)

        print("======")
        print(f"Query: {query}")
        for score, idx in zip(*result):
            print(f"{score.item():.4f} {samples[idx]}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)