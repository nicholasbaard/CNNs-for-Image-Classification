import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_scalars(logdir, tag):
    ea = EventAccumulator(logdir, size_guidance={'scalars': 0})
    ea.Reload()

    if tag not in ea.Tags()['scalars']:
        raise ValueError(f"Tag {tag} not found in logs")

    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    
    return steps, values

def plot_scalars(logdir, tag):
    tags = ['Loss/train', 'Accuracy/train', 'Loss/test', 'Accuracy/test']

    steps_lt, values_lt = extract_scalars(logdir, tags[0])
    steps_at, values_at = extract_scalars(logdir, tags[1])
    steps_ltest, values_ltest = extract_scalars(logdir, tags[2])
    steps_atest, values_atest = extract_scalars(logdir, tags[3])

    # Plot using matplotlib
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    ax[0, 0].plot(steps_lt, values_lt, label=tags[0])
    ax[0, 0].set_xlabel('Step')
    ax[0, 0].set_ylabel('Value')
    ax[0, 0].set_title(f'{tags[0]}')
    # ax[0, 0].legend()
    ax[0, 0].grid(True)

    ax[0, 1].plot(steps_at, values_at, label=tags[1])
    ax[0, 1].set_xlabel('Step')
    ax[0, 1].set_ylabel('Value')
    ax[0, 1].set_title(f'{tags[1]}')
    # ax[0, 1].legend()
    ax[0, 1].grid(True)

    ax[1, 0].plot(steps_ltest, values_ltest, label=tags[2])
    ax[1, 0].set_xlabel('Step')
    ax[1, 0].set_ylabel('Value')
    ax[1, 0].set_title(f'{tags[2]}')
    # ax[1, 0].legend()
    ax[1, 0].grid(True)

    ax[1, 1].plot(steps_atest, values_atest, label=tags[3])
    ax[1, 1].set_xlabel('Step')
    ax[1, 1].set_ylabel('Value')
    ax[1, 1].set_title(f'{tags[3]}')
    # ax[1, 1].legend()
    ax[1, 1].grid(True)
    plt.tight_layout()
    plt.show()
    
    fig.savefig(f'../plots/training_results.png')

logdir = 'runs/May28_10-09-24_nic-notebook'  # Replace with your log directory
tag = 'Loss/train'  # e.g., 'loss', 'accuracy'

plot_scalars(logdir, tag)
