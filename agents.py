import torch
from environments import DroneLidar

import random


class ParticleFilterAgent:
    def __init__(
        self,
        environment: DroneLidar,
        weighting_function: callable,
        position_estimation_function: callable,
        num_particles: int = 100,
        keep_proportion: float = 0.5,
        fuzzing_std: float = 1.0,
        weighing_function_arguments: dict = {},
    ):
        self.environment = environment
        self.weighting_function = weighting_function
        self.weighting_function_arguments = weighing_function_arguments
        self.position_estimation_function = position_estimation_function
        self.num_particles = num_particles
        self.particles = (
            torch.rand((num_particles, environment.dimensions), device=self.environment.device)
            * self.environment.max_distance
            - self.environment.max_distance / 2
        )
        self.fuzzing_std = fuzzing_std
        self.keep_proportion = keep_proportion
        self.noise_std = fuzzing_std
        self.previous_action = (0,) * self.environment.dimensions
        self.estimated_position = None

    def reset(self):
        self.particles = (
            torch.rand((self.num_particles, self.environment.dimensions), device=self.environment.device)
            * self.environment.max_distance
            - self.environment.max_distance / 2
        )

    def get_action(self, observation: tuple[torch.Tensor, torch.Tensor]) -> tuple[float, ...]:
        """
        :param observation: tuple of (landmarks, lidar measurements to landmarks)
        :returns: action to take
        """
        landmarks, lidar_readings = observation

        self.transition_particles()

        weights = self.weighting_function(self.particles, landmarks, lidar_readings, self.weighting_function_arguments)

        self.estimated_position = self.position_estimation_function(self.particles, weights)

        self.resample_particles(weights)

        # Non random actions
        # self.previous_action = self.move_towards_goal(self.estimated_position, landmarks[0])
       
        # Random actions
        if self.environment.dimensions == 3:
            self.previous_action = (random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1))
        elif self.environment.dimensions == 2:
            self.previous_action = (random.uniform(-1, 1), random.uniform(-1, 1))

        return self.previous_action

    def resample_particles(self, weights):
        _, indices = torch.sort(weights, descending=True)
        n_replace = int(self.num_particles * (1 - self.keep_proportion))
        self.particles[indices[-n_replace:]] = (
            self.particles[indices[:n_replace]]
            + torch.randn((n_replace, self.environment.dimensions), device=self.environment.device) * self.fuzzing_std
        )

    def transition_particles(self):
        self.particles += torch.tensor(self.previous_action, device=self.environment.device)
        self.particles = torch.clamp(
            self.particles, -self.environment.max_distance / 2, self.environment.max_distance / 2
        )

    def move_towards_goal(self, estimated_position: torch.Tensor, goal_position: torch.Tensor) -> tuple:
        """
        Just move towards the goal.
        """
        direction = goal_position - estimated_position
        direction_normalized = direction / torch.norm(direction)
        return tuple(direction_normalized.cpu())


def weight_particles_gaussian_pdf(particles, landmarks, lidar_readings, weighting_function_arguments):
    assert (
        "noise_std" in weighting_function_arguments
    ), "When using this weighting function, you must specify a noise_std in the weighting_function_arguments dict."
    weights = gaussian_pdf(
        torch.sum(
            torch.nan_to_num(
                torch.norm(particles[:, None, :] - landmarks[None, :, :], dim=-1) - lidar_readings[None, :], 0
            )
            ** 2,
            dim=-1,
        ),
        std=weighting_function_arguments["noise_std"],
    )

    if torch.sum(weights) == 0:
        weights = torch.ones_like(weights)
    weights /= torch.sum(weights)
    return weights


def gaussian_pdf(x, mean: float = 0, std: float = 1) -> torch.Tensor:
    """
    :returns: The Gaussian PDF evaluated at x.

    :param x: points to evalute at.
    :param mean: the mean of the Gaussian distribution.
    :param std: the standard deviation of the Gaussian distribution.
    """
    if not isinstance(std, torch.Tensor):
        std = torch.tensor(std, dtype=x.dtype, device=x.device)
    return torch.exp(
        -(0.5 * torch.log(2 * torch.tensor([torch.pi], dtype=x.dtype, device=x.device)))
        - torch.log(std)
        - 0.5 * ((x - mean) / std) ** 2
    )


def weight_particles_softmax_dot_product(particles, landmarks, lidar_readings, weighting_function_arguments: dict):
    return torch.softmax(
        ((torch.norm(particles[:, None, :] - landmarks[None, :, :], dim=-1)) @ torch.nan_to_num(lidar_readings, 0))
        * weighting_function_arguments.get("sharpness", 1),
        dim=0,
    )


def top_few_weighted_mean(particles, weights, few: int = 1):
    _, indices = torch.sort(weights, descending=True)
    return (particles[indices[:few]] * weights[indices[:few]][:, None]).mean(0)


def weighted_mean(particles, weights):
    return top_few_weighted_mean(particles, weights, few=len(particles))
