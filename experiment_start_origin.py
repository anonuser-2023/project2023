import torch
import tqdm

import agents
import environments

import matplotlib.pyplot as plt

import time
import csv

import numpy as np
import pandas as pd

from collections import defaultdict

def main():

    # Code for plotting individual histograms:
    results = {}
    for dim in [2, 3]:
        for nan_prob in [0.1, 0.3, 0.5]:
            # testing different noise_std values
            for noise_std in [2.0]:
                key = f'{dim} dimensions, nan_probability = {nan_prob}, gaussian_pdf weighting, noise_std = {noise_std}'
                results[key] = try_something(dim, nan_prob, agents.weight_particles_gaussian_pdf, {"noise_std": noise_std})
            
            # softmax_dot_product weighting with different values for sharpness
            for sharpness in [0.125, 0.25, 0.5, 1.0]:
                key = f'{dim} dimensions, nan_probability = {nan_prob}, softmax_dot_product weighting, sharpness = {sharpness}'
                results[key] = try_something(dim, nan_prob, agents.weight_particles_softmax_dot_product, {"sharpness": sharpness})

    for key, result in results.items():
        fig, ax = plt.subplots(figsize=(10, 6))

        color = 'orange' if 'gaussian_pdf' in key else 'blue'
    
        ax.hist(result, bins=30, color=color)
        ax.set_title(f"{key}, 1000 step limit")

        ax.set_xlabel('Distance')
        ax.set_ylabel('Frequency')
        fig.savefig(f'./individual_histograms/{key}.pdf', format='pdf')
    
    # code for plotting overlapping histograms
    specific_cases = {}
    for dim in [2, 3]:
        for nan_prob in [0.1, 0.3, 0.5]:
            key_gaussian = f'{dim} dimensions, nan_probability = {nan_prob}, gaussian_pdf weighting, noise_std = 2.0'
            key_softmax = f'{dim} dimensions, nan_probability = {nan_prob}, softmax_dot_product weighting, sharpness = 0.125'
            specific_cases[key_gaussian] = results[key_gaussian]
            specific_cases[key_softmax] = results[key_softmax]

    for dim in [2, 3]:
        for nan_prob in [0.1, 0.3, 0.5]:
            fig, axs = plt.subplots(1)

            key_gaussian = f'{dim} dimensions, nan_probability = {nan_prob}, gaussian_pdf weighting, noise_std = 2.0'
            key_softmax = f'{dim} dimensions, nan_probability = {nan_prob}, softmax_dot_product weighting, sharpness = 0.125'
            
            axs.hist(specific_cases[key_gaussian], bins=50, color='blue', alpha=0.5, label='Gaussian PDF')
            axs.hist(specific_cases[key_softmax], bins=50, color='orange', alpha=0.5, label='Attention')
            
            axs.set_title(f"{dim} Dimensions with Obstacle Probability={nan_prob}\nGaussian PDF vs Attention with Sharpness=0.125", fontsize=10)
            axs.set_xlabel('Distance')
            axs.set_ylabel('Frequency')
            
            axs.legend()

            plt.tight_layout()
            fig.savefig(f'./overlapping_histograms/dimension_{dim}_nan_probability_{nan_prob}_overlap.pdf', format='pdf')

    # What was here before:
    # try_something(3, 0.1, agents.weight_particles_gaussian_pdf, {"noise_std": 2.0})
    # try_something(3, 0.1, agents.weight_particles_softmax_dot_product, {"sharpness": 1.0})

def try_something(
    dimensions: int, nan_probability: float, weighting_function: callable, weighting_function_arguments=None, 
    num_particles=1000, num_landmarks=10, device="cpu", runs=10
):
    if weighting_function_arguments is None:
        weighting_function_arguments = {}

    torch.manual_seed(0)

    environment = environments.DroneLidar(
        dimensions,
        noise_std=2.0,  # default 2
        device=device,
        num_landmarks=num_landmarks,
        max_distance=100.0,
        nan_probability=nan_probability,
    )
    agent = agents.ParticleFilterAgent(
        environment=environment,
        weighting_function=weighting_function,
        weighing_function_arguments=weighting_function_arguments,
        position_estimation_function=agents.weighted_mean,
        num_particles=num_particles,
        keep_proportion=0.9,
    )

    returns = []

    # For plotting histograms
    distances = []
    
    # Added for agent test
    # successful_trials = 0
    # successful_run_steps = []
    # overall_steps = []

    # Time for each step
    times = []

    for _ in tqdm.tqdm(range(runs)):
        agent.reset()
        observation = environment.reset()
        total_reward = 0
        done = False
        while not done:
            start = time.perf_counter()
            action = agent.get_action(observation)
            observation, reward, done, info = environment.step(action)
            end = time.perf_counter()

            times.append(end - start)

            total_reward += reward

            # Added for distances
            estimated_position = agent.estimated_position
            distances.append(torch.norm(estimated_position - environment.drone_position))

            # Added for agent test (got to first landmark)
            # if done and info['reached_goal']:
            #     successful_trials += 1
            #     successful_run_steps.append(info['steps_taken'])
            
            # overall_steps.append(info['steps_taken'])

        returns.append(total_reward)

    # print(returns)

    # for histograms
    # return distances
    # for time per step
    return times
    # Added for testing agent
    # prob_success = successful_trials / runs
    # avg_steps = sum(successful_run_steps) / len(successful_run_steps) if successful_run_steps else None
    # overall_avg = sum(overall_steps) / len(overall_steps)
    # return prob_success, avg_steps, overall_avg


def timing(dimensions, num_particles, num_landmarks, weighting_functions, devices, gpu):
    # Run a warmup run
    warmup_run = try_something(dimensions=2, nan_probability=0.1, weighting_function=weighting_functions[0], 
                                      num_particles=100, num_landmarks=10, device=devices[0],
                                      weighting_function_arguments={"noise_std": 2.0})
    
    results = []
    all_times = defaultdict(list)

    for device in devices:
        for dim in dimensions:
            for weighting_func in weighting_functions:
                if weighting_func.__name__ == 'weight_particles_gaussian_pdf':
                    weighting_function_argument = {"noise_std": 2.0}
                    name = 'Gaussian PDF'
                else:
                    weighting_function_argument = {"sharpness": 1.0}
                    name = 'Attention'
            
                # Test number of particles
                for num_particle in num_particles:
                    times = try_something(dimensions=dim, nan_probability=0.1, weighting_function=weighting_func, 
                                  num_particles=num_particle, num_landmarks=10, device=device,
                                  weighting_function_arguments=weighting_function_argument)
                    
                    config = f'{dim}, 0.1, {num_particle}, 10, {device}, {name}'
                    all_times[config].append(times)

                    mean_time = np.mean(times)
                    std_time = np.std(times)
                    results.append((dim, 0.1, num_particle, 10, weighting_func.__name__, device, mean_time, std_time))
                
                # Test number of landmarks (If we end up testing time for different landmarks)
                # for num_landmark in num_landmarks:
                #     times = try_something(dimensions=dim, nan_probability=0.1, weighting_function=weighting_func, 
                #                   num_particles=1000, num_landmarks=num_landmark, device=device,
                #                   weighting_function_arguments=weighting_function_argument)
                    
                #     config = f'{dim}, 0.1, 1000, {num_landmark}, {device}, {name}'
                #     all_times[config].append(times)

                #     mean_time = np.mean(times)
                #     std_dev_time = np.std(times)
                #     results.append((dim, 0.1, 1000, num_landmark, weighting_func.__name__, device, mean_time, std_dev_time))

    # Get the overall mean step time and std for each config
    final = defaultdict(list)
    for config in all_times:
        overall_mean = np.mean(all_times[config])
        overall_std_dev = np.std(all_times[config])
        final[config].append(overall_mean)
        final[config].append(overall_std_dev)

    # For different organization of csv file if needed
    # df = pd.DataFrame.from_dict(final, orient='index')
    # df.columns = ["Mean Time", "Standard Deviation"]
    # df.reset_index(inplace=True)
    # df.rename(columns={'index': 'Configuration'}, inplace=True)
    # df.to_csv('final_table.csv', index=False)

    df = pd.DataFrame.from_dict(final, orient='index')
    df.columns = ['Mean Time', 'Standard Deviation']
    df.reset_index(inplace=True)
    df[['Dimensions', 'Obstacle Probability', 'Num Particles', 'Num Landmarks', 'Device', 'Weighting Function']] = df['index'].str.split(', ', expand=True)
    df.drop(columns=['index'], inplace=True)
    df = df[['Dimensions', 'Obstacle Probability', 'Num Particles', 'Num Landmarks', 'Weighting Function', 'Device', 'Mean Time', 'Standard Deviation']]
    df.to_csv(f'timings_{gpu}.csv', index=False)

def plot_timing(dimension, file, gpu_name, param_values, cpu_name):
    """
    dimension is the dimension we want to plot for
    """
    df = pd.read_csv(file)

    df['Num Particles'] = df['Num Particles'].astype(int)
    df['Dimensions'] = df['Dimensions'].astype(int)

    filtered_df = df[df['Dimensions'] == dimension]

    devices = filtered_df['Device'].unique()
    weighting_functions = filtered_df['Weighting Function'].unique()

    # plt.figure(figsize=(10, 6))
    # Create a new figure for this dimension
    fig, ax = plt.subplots(figsize=(6, 3))

    for device in devices:
        for weighting_function in weighting_functions:
            # Filter the data based on the current combination of device and weighting function
            sub_df = filtered_df[(filtered_df['Device'] == device) & (filtered_df['Weighting Function'] == weighting_function)]
            
            if sub_df.empty:
                continue

            # For no error bars
            # plt.plot([str(x) for x in param_values], sub_df['Mean Time'], 'o-', label=f'{weighting_function} {device}')
            
            # For error bars
            plt.errorbar([str(x) for x in param_values], sub_df['Mean Time'], yerr=sub_df['Standard Deviation'], marker='o', linestyle='-', label=f'{weighting_function} {device}', capsize=3, elinewidth=1, alpha=0.7)
            
    plt.title(f'Dimensions: {dimension}, GPU: {gpu_name}, CPU: {cpu_name}')
    plt.xlabel('Num Particles')
    plt.ylabel('Mean Time per Step')
    plt.legend(loc='upper left', fontsize=6)

    plt.tight_layout()
    plt.savefig(f'num_particles_{dimension}_dimensions_{gpu_name}.pdf', format='pdf')

if __name__ == "__main__":

    if torch.cuda.is_available():
        print("GPU is available")
    else:
        print("GPU is not available")

    dimensions = [2, 3]
    num_particles = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    num_landmarks = [10]
    devices = ["cuda", "cpu"]
    weighting_functions = [agents.weight_particles_gaussian_pdf, agents.weight_particles_softmax_dot_product]

    # timing(dimensions, num_particles, num_landmarks, weighting_functions, devices, '2060')

    plot_timing(2, 'timings_3090.csv', 3090, num_particles, 'Ryzen 5 5600X')
    plot_timing(3, 'timings_3090.csv', 3090, num_particles, 'Ryzen 5 5600X')
    plot_timing(2, 'timings_2060.csv', 2060, num_particles, 'Intel i5-10400')
    plot_timing(3, 'timings_2060.csv', 2060, num_particles, 'Intel i5-10400')
    
    # For plotting histograms
    # main()

    # For testing task 3 (agent RL task)
    # for weighting_func in weighting_functions:
    #     if weighting_func.__name__ == 'weight_particles_gaussian_pdf':
    #         name = 'weight_particles_gaussian_pdf'
    #         weighting_function_argument = {"noise_std": 8.0}
    #     else:
    #         name = 'weight_particles_softmax_dot_product'
    #         weighting_function_argument = {"sharpness": 1.0}

    #     prob_success, avg_steps, ovr_avg = try_something(
    #     dimensions=3, 
    #     nan_probability=0.1, 
    #     weighting_function=weighting_func, 
    #     weighting_function_arguments=weighting_function_argument, 
    #     num_particles=100, 
    #     num_landmarks=5, 
    #     device="cuda",
    #     runs=100)

    #     print(f"{name} results:")
    #     print(f"Probability of success: {prob_success}")
    #     print(f"Successful Average steps: {avg_steps}")
    #     print(f"Overall Average steps: {ovr_avg}")



