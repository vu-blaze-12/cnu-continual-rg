from rg_dataset import make_rg_split

if __name__ == "__main__":
    split = make_rg_split("leg_counting", n_train=10, n_val=5, seed=42)

    print("\nSample formatted training example:")
    print(split["train_texts"][0])

    print("\nSample validation prompt:")
    print(split["val_prompts"][0])

    print("\nSample validation answer:")
    print(split["val_answers"][0])
