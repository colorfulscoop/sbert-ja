from sentence_transformers import SentenceTransformer, util
import torch


def main(model, top_k=3):
    samples = [
        "外をランニングするのが好きです",
        "海外旅行に行くのが趣味です",
        "天ぷらが食べたい",
        "りんごが食べたい",
        "揚げ物は食べたくない",
        "今日は天気がいいです",
    ]
    model = SentenceTransformer(model)
    samples_embedding = model.encode(samples, convert_to_tensor=True)

    # Query sentences:
    queries = [
        "ランニングは好きです",
        "旅行は海外に行きたい",
        "揚げ物が食べたい",
    ]

    top_k = min(top_k, len(samples))
    for query in queries:
        query_embedding = model.encode(query, convert_to_tensor=True)

        scores = util.pytorch_cos_sim(query_embedding, samples_embedding)[0]
        result = torch.topk(scores, k=top_k)

        print("======")
        print(f"Query: {query}")
        for score, idx in zip(*result):
            print(score, samples[idx])


if __name__ == "__main__":
    import fire

    fire.Fire(main)