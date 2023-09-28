import itertools
import pathlib

import torch
import tqdm
import matplotlib.pyplot as plt

import agents
import environments


def main():
    try_something(
        2,
        0.1,
        agents.weight_particles_gaussian_pdf,
        {"noise_std": 5.0},
        position_estimation_function=agents.weighted_mean,
    )
    try_something(
        2, 0.1, agents.weight_particles_softmax_dot_product, position_estimation_function=agents.weighted_mean
    )
    try_something(
        2,
        0.1,
        agents.weight_particles_gaussian_pdf,
        {"noise_std": 5.0},
        position_estimation_function=agents.top_few_weighted_mean,
    )
    try_something(
        2, 0.1, agents.weight_particles_softmax_dot_product, position_estimation_function=agents.top_few_weighted_mean
    )


def try_something(
    dimensions: int,
    nan_probability: float,
    weighting_function: callable,
    weighting_function_arguments=None,
    position_estimation_function=agents.weighted_mean,
):
    if weighting_function_arguments is None:
        weighting_function_arguments = {}
    torch.manual_seed(0)
    save_directory = pathlib.Path("results") / "uniform"
    if not save_directory.exists():
        save_directory.mkdir(parents=True)

    environment = environments.DroneLidar(
        dimensions,
        noise_std=5.0,
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_landmarks=10,
        max_distance=100.0,
        nan_probability=nan_probability,
    )
    agent = agents.ParticleFilterAgent(
        environment=environment,
        weighting_function=weighting_function,
        weighing_function_arguments=weighting_function_arguments,
        position_estimation_function=position_estimation_function,
        num_particles=100,
        keep_proportion=0.9,
    )

    size = 51
    halfway = size // 2
    distances = torch.empty((size,) * dimensions, device=environment.device)

    for starting_position in tqdm.tqdm(itertools.product(range(size), repeat=dimensions), total=size**dimensions):
        agent.reset()
        environment.drone_position = (
            torch.tensor(starting_position, dtype=torch.float32, device=environment.device) - halfway
        )
        landmarks, lidar_readings = environment._observe()
        weights = agent.weighting_function(
            agent.particles, landmarks, lidar_readings, agent.weighting_function_arguments
        )
        agent.resample_particles(weights)
        agent.estimated_position = (agent.particles * weights[:, None]).mean(0)
        estimated_position = agent.estimated_position
        distances[starting_position] = torch.norm(estimated_position - environment.drone_position)

    if dimensions == 2:
        plt.clf()
        plt.imshow(distances.cpu().numpy())
        plt.colorbar()
        plt.savefig(save_directory / f"{weighting_function.__name__}_{position_estimation_function.__name__}.png")
        plt.close()


if __name__ == "__main__":
    main()
