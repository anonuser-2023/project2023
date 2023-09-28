import pathlib

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch
import torch.nn as nn
import numpy as np
import tqdm

import agents


def main():
    data_path = pathlib.Path("data")
    save_path = pathlib.Path("results") / "explain"

    # Scalar
    n_particles = 64
    x = torch.linspace(-2, 2, n_particles)
    y = agents.gaussian_pdf(x, 0)
    y /= y.sum()

    plt.figure(figsize=(6, 3))
    ax1 = plt.gca()
    ax1.plot(x, y, color="tab:blue")
    ax1.set_xlabel("difference from 1")
    ax1.set_ylabel("Gaussian PDF weights")

    ax2 = ax1.twinx()
    ax2.plot(x, torch.softmax(x, 0), color="tab:orange")
    ax2.set_ylabel("attention weights")

    plt.legend(
        handles=[
            Line2D([0], [0], color="tab:blue", label="Gaussian PDF"),
            Line2D([0], [0], color="tab:orange", label="attention"),
        ]
    )

    plt.title(f"Weighting Similarity to 1 ({n_particles} Particles)")
    plt.tight_layout()

    # plt.savefig(save_path / "gaussian_vs_attention_1d.png", dpi=600)
    plt.savefig(save_path / "gaussian_vs_attention_1d.pdf")
    plt.close()

    # Pretrained embeddings

    embedding_layer = prepare_embedding_layer(data_path / "glove.6B.100d.txt")
    embedding_weights = embedding_layer.weight.detach().clone()
    some_embedding = embedding_weights[0]
    sum_differences = ((embedding_weights - some_embedding[None, :])).sum(1)

    gaussian_pdf_weights = agents.gaussian_pdf(sum_differences)
    gaussian_pdf_weights /= gaussian_pdf_weights.sum()
    sorted_sum_differences, indices = sum_differences.sort()
    sorted_gaussian_weights = gaussian_pdf_weights[indices]

    sharpness = 1 / 4
    attention_weights = torch.softmax((embedding_weights @ some_embedding) * sharpness, 0)
    sorted_attention_weights = attention_weights[indices]

    plt.figure(figsize=(6, 3))

    ax1 = plt.gca()
    ax1.plot(sorted_sum_differences, sorted_gaussian_weights, color="tab:blue")
    ax1.set_xlabel("sum of differences")
    ax1.set_ylabel(f"Gaussian PDF weights")

    ax2 = ax1.twinx()
    ax2.plot(sorted_sum_differences, sorted_attention_weights, color="tab:orange")
    ax2.set_ylabel("attention weights")

    plt.legend(
        handles=[
            Line2D([0], [0], color="tab:blue", label="Gaussian PDF"),
            Line2D([0], [0], color="tab:orange", label=f"attention (sharpness={sharpness})"),
        ]
    )

    plt.title(f"Similarity to {embedding_weights.shape[0]} {embedding_weights.shape[1]}D embeddings")
    plt.tight_layout()

    # plt.savefig(save_path / f"gaussian_vs_attention_pretrained.png", dpi=600)
    plt.savefig(save_path / f"gaussian_vs_attention_pretrained.pdf")


def prepare_embedding_layer(path: pathlib.Path, word2idx: dict = None) -> nn.Embedding:
    saved_embeddings_path = path.parent / f"{path.stem}_embeddings.pt"
    if saved_embeddings_path.exists():
        return nn.Embedding.from_pretrained(torch.load(saved_embeddings_path))

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find glove file at {path}. Download embeddings from https://nlp.stanford.edu/data/glove.6B.zip and unzip it there."
        )
    with open(path, "r") as f:
        lines = f.readlines()

    tokens = set()
    word_vecs = {}
    for line in tqdm.tqdm(lines, desc="reading glove file"):
        line = line.strip().split()
        word = line[0]
        vec = np.array(line[1:], dtype=np.float32)

        if word2idx is None or word in word2idx.keys():
            word_vecs[word] = vec
            tokens.add(word)

    vocab_size = len(word_vecs)
    emb_dim = len(next(iter(word_vecs.values())))
    embedding = torch.zeros(vocab_size, emb_dim)

    for word, vector in tqdm.tqdm(word_vecs.items(), desc="preparing embedding layer"):
        if word2idx is not None:
            embedding[word2idx[word]] = torch.FloatTensor(vector)
        else:
            embedding[list(word_vecs.keys()).index(word)] = torch.FloatTensor(vector)

    torch.save(embedding, saved_embeddings_path)
    return nn.Embedding.from_pretrained(embedding)


if __name__ == "__main__":
    main()
