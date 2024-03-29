from typing import Any, Dict, List, Optional, Tuple, Union
from copy import deepcopy
from dataclasses import asdict, dataclass
import os
import argparse
import math
from pathlib import Path
import random
import uuid
import d4rl
import gym
import numpy as np
import pyrallis
import torch
from torch.distributions import Normal, TanhTransform, TransformedDistribution
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from ex import ContinuousCQL
from ex import TanhGaussianPolicy
from ex import FullyConnectedQFunction

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()

class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        return self.constant
envname='walker2d'
type='medium'
k=20000
beta=1
class TrainConfig:
    # Experiment
    device: str = "cuda"
    env: str = envname+"-"+type+"-v2"  # OpenAI gym environment name
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_freq: int = int(5e3)  # How often (time steps) we evaluate
    n_episodes: int = 10  # How many episodes run during evaluation
    max_timesteps: int = int(1e6)  # Max time steps to run environment
    checkpoints_path: Optional[str] = '/home/shenjiahao/desktop/rl_results'  # Save path
    load_model: str = envname+"/data/"+type+"/cql.pt"  # Model load file name, "" doesn't load
    # CQL
    buffer_size: int = 2_000_000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    alpha_multiplier: float = 1.0  # Multiplier for alpha in loss
    use_automatic_entropy_tuning: bool = True  # Tune entropy
    backup_entropy: bool = False  # Use backup entropy
    policy_lr: float = 3e-5  # Policy learning rate
    qf_lr: float = 3e-4  # Critics learning rate
    soft_target_update_rate: float = 5e-3  # Target network update rate
    bc_steps: int = int(0)  # Number of BC steps at start
    target_update_period: int = 1  # Frequency of target nets updates
    cql_n_actions: int = 10  # Number of sampled actions
    cql_importance_sample: bool = True  # Use importance sampling
    cql_lagrange: bool = False  # Use Lagrange version of CQL
    cql_target_action_gap: float = -1.0  # Action gap
    cql_temp: float = 1.0  # CQL temperature
    cql_min_q_weight: float = 10.0  # Minimal Q weight
    cql_max_target_backup: bool = False  # Use max target backup
    cql_clip_diff_min: float = -np.inf  # Q-function lower loss clipping
    cql_clip_diff_max: float = np.inf  # Q-function upper loss clipping
    orthogonal_init: bool = True  # Orthogonal initialization
    normalize: bool = True  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    # Wandb logging
    project: str = "CORL"
    group: str = "CQL-D4RL"
    name: str = "CQL"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)

TensorBatch = List[torch.Tensor]
def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)

def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std
def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env
class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def add_transition(self):
        # Use this method to add new data into the replay buffer during fine-tuning.
        # I left it unimplemented since now we do not do fine-tuning.
        raise NotImplementedError

    def sampobs(self, batch: TensorBatch) -> Dict[str, float]:
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        ) = batch
        return observations

cuda = True if torch.cuda.is_available() else False

class Generator(nn.Module):
    def __init__(self,state_dim,z_dim,hidden_dim):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(state_dim+z_dim,hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim,int(hidden_dim/2)),
            nn.BatchNorm1d(int(hidden_dim/2)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim/2),state_dim)
        )

    def forward(self, state,noise):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((state, noise), -1)
        g_state = self.model(gen_input)

        return g_state


class Discriminator(nn.Module):
    def __init__(self,state_dim,hidden_dim):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)

        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(hidden_dim, int(hidden_dim/2)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(int(hidden_dim/2), 1),
        )

    def forward(self, state):
        # Concatenate label embedding and image to produce input
        validity = self.model(state)
        return validity
config=TrainConfig()
env = gym.make(config.env)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
hidden_dim=100
z_dim=50
# Loss functions
adversarial_loss = torch.nn.MSELoss()

# Initialize generator and discriminator
generator = Generator(state_dim,z_dim,hidden_dim)
discriminator = Discriminator(state_dim,hidden_dim)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1,opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1,opt.b2))
optimizer_A=torch.optim.Adam(generator.parameters(),lr=0.01,betas=(opt.b1,opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
dataset = d4rl.qlearning_dataset(env)
state_mean, state_std = 0, 1
dataset["observations"] = normalize_states(
    dataset["observations"], state_mean, state_std
)
dataset["next_observations"] = normalize_states(
    dataset["next_observations"], state_mean, state_std
)
env = wrap_env(env, state_mean=state_mean, state_std=state_std)
replay_buffer = ReplayBuffer(
    state_dim,
    action_dim,
    10000000,
    "cuda",
)
replay_buffer.load_d4rl_dataset(dataset)
max_action = float(env.action_space.high[0])
seed = 0
set_seed(seed, env)

critic_1 = FullyConnectedQFunction(state_dim, action_dim, config.orthogonal_init).to(
    config.device
)
critic_2 = FullyConnectedQFunction(state_dim, action_dim, config.orthogonal_init).to(
    config.device
)
critic_1_optimizer = torch.optim.Adam(list(critic_1.parameters()), config.qf_lr)
critic_2_optimizer = torch.optim.Adam(list(critic_2.parameters()), config.qf_lr)

actor = TanhGaussianPolicy(
    state_dim, action_dim, max_action, orthogonal_init=config.orthogonal_init
).to(config.device)
actor_optimizer = torch.optim.Adam(actor.parameters(), config.policy_lr)
kwargs = {
    "critic_1": critic_1,
    "critic_2": critic_2,
    "critic_1_optimizer": critic_1_optimizer,
    "critic_2_optimizer": critic_2_optimizer,
    "actor": actor,
    "actor_optimizer": actor_optimizer,
    "discount": config.discount,
    "soft_target_update_rate": config.soft_target_update_rate,
    "device": config.device,
    # CQL
    "target_entropy": -np.prod(env.action_space.shape).item(),
    "alpha_multiplier": config.alpha_multiplier,
    "use_automatic_entropy_tuning": config.use_automatic_entropy_tuning,
    "backup_entropy": config.backup_entropy,
    "policy_lr": config.policy_lr,
    "qf_lr": config.qf_lr,
    "bc_steps": config.bc_steps,
    "target_update_period": config.target_update_period,
    "cql_n_actions": config.cql_n_actions,
    "cql_importance_sample": config.cql_importance_sample,
    "cql_lagrange": config.cql_lagrange,
    "cql_target_action_gap": config.cql_target_action_gap,
    "cql_temp": config.cql_temp,
    "cql_min_q_weight": config.cql_min_q_weight,
    "cql_max_target_backup": config.cql_max_target_backup,
    "cql_clip_diff_min": config.cql_clip_diff_min,
    "cql_clip_diff_max": config.cql_clip_diff_max,
}
cql =ContinuousCQL(**kwargs)
policy_file = Path(config.load_model)
print(policy_file)
cql.load_state_dict(torch.load(policy_file))
actor = cql.actor.to('cuda')
i=0
resd_g=[]
resd_a=[]
resa_g=[]
for t in range(int(k)):
        i+=100
        batch_size = 100
        batch = replay_buffer.sample(batch_size)
        batch = [b.to('cuda') for b in batch]
        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_states = replay_buffer.sampobs(batch)
        #labels = Variable(labels.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, 50))))

        # Generate a batch of images
        gen_states = generator(real_states, z)
        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_states)
        g_loss = adversarial_loss(validity, valid)
        #print(torch.mean(probs))
        #print(g_loss)
        true_loss=g_loss
        true_loss.backward()
        optimizer_G.step()

        optimizer_A.zero_grad()

        gen_states = generator(real_states, z)
        gen_acts, probs = actor.actgpu(gen_states)
        probs=probs/100
        actor_loss=adversarial_loss(probs,fake)
        a_loss=beta*actor_loss
        a_loss.backward()
        optimizer_A.step()
        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        validity_real = discriminator(real_states)
        d_real_loss = adversarial_loss(validity_real, valid)

        # Loss for fake images
        validity_fake = discriminator(gen_states.detach())
        d_fake_loss = adversarial_loss(validity_fake, fake)
        #print(d_real_loss)
        #print(d_fake_loss)
        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2
        #print(d_loss)
        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [A loss : %f]"
            % (t, k, i, 100000, d_loss.item(), g_loss.item(),a_loss.item())
        )
#torch.save(generator.state_dict(), envname+'/data/'+type+'/OODgenerator'+str(k)+'-optimizera-'+str(beta)+'.pth')
#torch.save(discriminator.state_dict(), envname+'/data/'+type+'/OODdiscriminator'+str(k)+'-optimizera-'+str(beta)+'.pth')