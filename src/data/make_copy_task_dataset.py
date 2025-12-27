import random

def make_copy_samples(n=500, seed=42, min_len=6, max_len=12):
    random.seed(seed)
    vocab = [
        "hello", "world", "foo", "bar", "baz", "test", "apple", "banana",
        "neuron", "network", "continual", "learning", "spider", "legs", "count"
    ]
    samples = []
    for _ in range(n):
        text_len = random.randint(min_len, max_len)
        text = " ".join(random.choices(vocab, k=text_len))
        sample = {
            "question": f"Copy this → {text}",
            "answer": text,
        }
        samples.append(sample)
    return samples
