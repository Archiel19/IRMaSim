import argparse as ap
import matplotlib.pyplot as plt


def plot_losses(exp_dir: str):
    with open(f'{exp_dir}/losses.log', 'r') as l_f:
        losses_pairs = [tuple(map(float, lp.strip().split(','))) for lp in l_f.readlines()]

    if len(losses_pairs[0]) == 3:
        losses = ['pi', 'v', 'total']
    elif len(losses_pairs[0]) == 2:
        losses = ['pi', 'v']
    else:
        losses = ['global']
    for i in range(len(losses)):
        plt.plot([l[i] for l in losses_pairs])
        plt.savefig(f'{exp_dir}/{losses[i]}_loss.png')
        plt.clf()


def plot_rewards(exp_dir: str):
    with open(f'{exp_dir}/rewards.log', 'r') as r_f:
        rews_pairs = [float(r.strip()) for r in r_f.readlines()]

    plt.plot([r for r in rews_pairs])
    plt.savefig(f'{exp_dir}/rewards.png')
    plt.clf()


def plot_resources(exp_dir: str):
    with open(f'{exp_dir}/resources.log', 'r') as r_f:
        lines = [tuple(rl.strip().split(',')) for rl in r_f.readlines()]
        indices = {name: idx for (idx, name) in enumerate(lines[0])}
        def get(line, tag):
            return line[indices[tag]]

        # Only plot the last run
        last_run = get(lines[-1], 'run')
        lines = [line for line in lines[1:] if line[indices['run']] == last_run]

        # Because we don't want to assume consecutive resource IDs
        first_id = get(lines[0], 'id')
        max_capacity = int(get(lines[0], 'cores'))
        chunk_size = 1
        for line in lines[1:]:
            if get(line, 'id') == first_id:
                break
            chunk_size += 1
            max_capacity += int(get(line, 'cores'))
        # TODO get the last trajectory only
        timestamps = [float(get(lines[i], 'time')) for i in range(0, len(lines), chunk_size)]
        loads = [sum(map(lambda l: int(get(l, 'busy_cores')), lines[i:i+chunk_size]))
                 for i in range(1, len(lines), chunk_size)]

    plt.plot(timestamps, loads)
    plt.axhline(y=max_capacity, linestyle='dashed', color='r')
    plt.savefig(f'{exp_dir}/resources.png')
    plt.clf()


def do_plots(args):
    if args.loss:
        plot_losses(args.exp_dir)
    if args.reward:
        plot_rewards(args.exp_dir)
    plot_resources(args.exp_dir)


if __name__ == '__main__':
    parser = ap.ArgumentParser('Creates some plots with the results on the log files')
    parser.add_argument('-d', '--exp-dir', type=str, default='')
    parser.add_argument('-l', '--loss', default=False, action=ap.BooleanOptionalAction)
    parser.add_argument('-r', '--reward', default=False, action=ap.BooleanOptionalAction)
    args = parser.parse_args()
    do_plots(args)
