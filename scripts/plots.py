import argparse
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def parse_args():
    parser = argparse.ArgumentParser(description="Plot mean and std from tensorboard logs.")
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task name (e.g. single, competitive, cooperative, teams)",
    )
    parser.add_argument(
        "--algo", type=str, default=None, help="Algorithm name (e.g. dqn, ppo, iql, mappo)"
    )
    parser.add_argument(
        "--policy", type=str, default=None, help="Policy name (e.g. cnn, attn, ctde_cnn)"
    )
    args = parser.parse_args()

    if not any([args.task, args.algo, args.policy]):
        parser.error("At least one of --task, --algo, or --policy must be provided.")

    return args


def extract_tb_data(exp_dir, tag_name="rollout/ep_rew_mean"):
    tb_base_dir = os.path.join(exp_dir, "tb")
    if not os.path.exists(tb_base_dir):
        return None, None

    # Find the directory containing the actual tfevents file
    tb_dir = None
    for root, _dirs, files in os.walk(tb_base_dir):
        if any("events.out.tfevents" in f for f in files):
            tb_dir = root
            break

    if not tb_dir:
        return None, None

    ea = EventAccumulator(tb_dir)
    ea.Reload()

    # Find the matching tag
    target_tag = None
    if tag_name in ea.Tags()["scalars"]:
        target_tag = tag_name
    else:
        for tag in ea.Tags()["scalars"]:
            if tag.endswith("ep_rew_mean"):
                target_tag = tag
                break

    if not target_tag:
        return None, None

    events = ea.Scalars(target_tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]

    return steps, values


def main():
    args = parse_args()

    exp_base_dir = "experiments"
    all_exps = [d for d in os.listdir(exp_base_dir) if os.path.isdir(os.path.join(exp_base_dir, d))]

    matched_exps = []
    for exp in all_exps:
        # Check if conditions match
        match = True
        if args.task and args.task not in exp:
            match = False
        if args.algo and args.algo not in exp:
            match = False
        if args.policy and args.policy not in exp:
            match = False

        if match:
            matched_exps.append(exp)

    if not matched_exps:
        print("No experiments found matching the criteria.")
        return

    print(f"Found {len(matched_exps)} matching experiments:")
    for m in matched_exps:
        print(f"  - {m}")

    # Group matching experiments by base name (removing seed suffix)
    grouped_exps = defaultdict(list)
    for exp in matched_exps:
        base_name = re.sub(r"_seed\d+_\d+$", "", exp)
        grouped_exps[base_name].append(exp)

    if not grouped_exps:
        print("No valid data could be found matching the criteria.")
        return

    # First pass: collect all data and find global minimum max_timestep
    group_data = {}
    global_max_step = float("inf")

    for base_name, exps in grouped_exps.items():
        all_values = []
        all_steps = []
        for exp in exps:
            exp_dir = os.path.join(exp_base_dir, exp)
            steps, values = extract_tb_data(exp_dir, "rollout/ep_rew_mean")

            if values is not None and len(steps) > 0:
                all_values.append(values)
                all_steps.append(steps)

                # Update the global minimum of the maximum reached timesteps
                exp_max_step = max(steps)
                if exp_max_step < global_max_step:
                    global_max_step = exp_max_step
            else:
                print(f"Warning: No valid ep_rew_mean data found in {exp_dir}")

        if not all_values:
            continue

        group_data[base_name] = {"values": all_values, "steps": all_steps}

    if not group_data:
        print("No valid data could be found matching the criteria.")
        return

    print(f"Clipping all groups to global max timestep: {global_max_step}")

    # Plotting setup
    plt.style.use("Solarize_Light2")
    plt.figure(figsize=(10, 6))

    plotted_any = False

    # Create a uniform x-axis for interpolating to align perfectly
    common_steps = np.linspace(0, global_max_step, 1000)

    for base_name, data in group_data.items():
        all_values = data["values"]
        all_steps = data["steps"]

        interp_values_list = []

        # Interpolate each seed's data to the uniform axis
        for b_steps, b_vals in zip(all_steps, all_values, strict=False):
            interp_vals = np.interp(common_steps, b_steps, b_vals)
            interp_values_list.append(interp_vals)

        if not interp_values_list:
            continue

        # Convert to numpy
        interpolated_values = np.array(interp_values_list)

        # Calculate mean and std
        mean_vals = np.mean(interpolated_values, axis=0)
        std_vals = np.std(interpolated_values, axis=0)

        # Plot line and shaded area
        line = plt.plot(common_steps, mean_vals, label=base_name, linewidth=2)[0]
        plt.fill_between(
            common_steps,
            mean_vals - std_vals,
            mean_vals + std_vals,
            color=line.get_color(),
            alpha=0.3,
        )
        plotted_any = True

    if not plotted_any:
        print("No valid data could be plotted.")
        return

    # Create an active title
    title_parts = []
    if args.task:
        title_parts.append(f"Task: {args.task}")
    if args.algo:
        title_parts.append(f"Algo: {args.algo}")
    if args.policy:
        title_parts.append(f"Policy: {args.policy}")

    plt.title("Episodic Reward Mean")
    plt.xlabel("Step / Iteration")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.tight_layout()

    # Save the plot
    out_file = f"plot_{'-'.join([s.replace(' ', '') for s in title_parts])}.png"
    plt.savefig(out_file, dpi=300)
    print(f"Plot saved to {out_file}")
    plt.show()


if __name__ == "__main__":
    main()
