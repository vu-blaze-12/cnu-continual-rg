import argparse
import reasoning_gym as rg
from reasoning_gym import get_score_answer_fn
from rich import print as rprint


def main(task_name: str, n_samples: int):
    dataset = rg.create_dataset(task_name)

    # Dataset size
    try:
        dataset_size = dataset.size
    except AttributeError:
        dataset_size = len(dataset)

    if n_samples > dataset_size:
        raise ValueError(f"Requested {n_samples} samples, but dataset only has {dataset_size}")

    # Fetch samples
    samples = [dataset[i] for i in range(n_samples)]

    # Print examples
    rprint(f"\n[bold cyan]Task:[/bold cyan] {task_name}")
    rprint(f"[bold cyan]Showing {min(3, n_samples)} examples[/bold cyan]")

    for i, sample in enumerate(samples[:3]):
        rprint(f"\n[bold]Example {i+1}[/bold]")
        rprint(f"[bold]Q:[/bold] {sample['question']}")
        rprint(f"[bold]A:[/bold] {sample['answer']}")

    # Verifier sanity check
    verifier = get_score_answer_fn(task_name)
    score = verifier(samples[0]["answer"], samples[0])

    rprint(f"\n[bold green]Verifier score (gold answer): {score}[/bold green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()

    main(args.task, args.n)
