import torch
import gymnasium as gym
from gymnasium import spaces


class DroneLidar(gym.Env):
    def __init__(
        self,
        dimensions: int = 2,
        noise_std: float = 0.0,
        device: str = "cpu",
        num_landmarks: int = 5,
        max_distance: float = 100.0,
        # default drone_distance is 1.0. Can change to make agent task harder
        done_distance: float = 1.0,
        step_cost: float = 1.0,
        # default step limit is 1000
        step_limit: int = 1000,
        nan_probability: float = 0.0,
    ):
        super(DroneLidar, self).__init__()
        self.device = torch.device(device)

        self.dimensions = dimensions
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(dimensions,), dtype=float)
        self.observation_space = spaces.Box(
            low=-max_distance / 2, high=max_distance / 2, shape=(dimensions + dimensions * num_landmarks,), dtype=float
        )

        self.step_cost = step_cost
        self.max_distance = max_distance
        self.done_distance = done_distance
        self.landmarks: torch.Tensor = torch.randint(
            -int(self.max_distance / 2), int(self.max_distance / 2) + 1, (num_landmarks, dimensions), device=self.device
        )
        self.nan_probability = nan_probability

        # The first landmark is the goal.
        self.noise_std = noise_std
        self.drone_position = None
        self.step_limit = step_limit
        self.steps_taken = 0

    def reset(self, start_position: tuple[float, ...] = None) -> torch.Tensor:
        if start_position is None:
            start_position = (0.0,) * self.dimensions
        self.landmarks: torch.Tensor = torch.randint(
            -int(self.max_distance / 2), int(self.max_distance / 2) + 1, self.landmarks.shape, device=self.device
        )
        self.drone_position = torch.tensor(start_position, dtype=torch.float32, device=self.device)
        self.steps_taken = 0
        return self._observe()

    def step(self, action: torch.Tensor) -> tuple:
        self.drone_position += torch.tensor(action, device=self.device)
        self.drone_position = torch.clamp(self.drone_position, -self.max_distance / 2, self.max_distance / 2)

        distance = torch.norm(self.landmarks[0] - self.drone_position, 2).item()
        reached_goal = distance <= self.done_distance
        
        # Disabling terminal state:
        done =  self.steps_taken >= self.step_limit # distance <= self.done_distance or self.steps_taken >= self.step_limit
        # Non disabled terminal state:
        # done = distance <= self.done_distance or self.steps_taken >= self.step_limit

        reward = reached_goal * self.max_distance - (not done) * self.step_cost
        self.steps_taken += 1

        return self._observe(), reward, done, {'reached_goal': reached_goal,
                                               'steps_taken': self.steps_taken}

    def _observe(self) -> tuple[torch.Tensor, torch.Tensor]:
        lidar_readings = torch.norm(self.landmarks - self.drone_position, 2, dim=1)
        if self.noise_std > 0:
            lidar_readings += torch.normal(0, self.noise_std, size=lidar_readings.shape, device=self.device)
        if self.nan_probability > 0:
            lidar_readings[torch.rand(lidar_readings.shape) < self.nan_probability] = torch.nan
        return self.landmarks, lidar_readings
    
    # For wifi strength experiment
    def _observe(self) -> tuple[torch.Tensor, torch.Tensor]:
        wifi_strength = (self.max_distance / torch.norm(self.landmarks - self.drone_position, 2, dim=1)) ** 2
        if self.noise_std > 0:
            wifi_strength += torch.normal(0, self.noise_std, size=wifi_strength.shape, device=self.device)
        if self.nan_probability > 0:
            wifi_strength[torch.rand(wifi_strength.shape) < self.nan_probability] = torch.nan
        return self.landmarks, wifi_strength

