"""Frozen-network opponent. `torch` is imported lazily, only when one is built.

The network is any callable that maps a batch of flattened observations,
float32 of shape (1, `OBS_DIM`), to logits of shape (1, 22, 4) or (1, 88). The
opponent takes the argmax per component, or samples from the softmax when
`deterministic=False`. Weights are never changed (eval mode, no gradients).

`FrozenNetworkOpponent.from_torchscript(path)` loads a `torch.jit` file, so a
checkpoint can be played without importing the code that trained it.
"""

import numpy as np

from letorifighter.bodies import NUM_CONTROLLABLES, NUM_JOINT_STATES

BODY_KEYS = ("positions", "velocities", "groin_height", "joint_states", "grips")


def flatten_observation(obs: dict) -> np.ndarray:
    """Concatenate a per-agent observation into one float32 vector (`OBS_DIM`)."""
    parts = [
        obs[body][key].ravel() for body in ("self", "opponent") for key in BODY_KEYS
    ]
    parts.append(np.asarray(obs["match_frame"]).ravel())
    parts.append(np.asarray(obs["frames_next_turn"]).ravel())
    return np.concatenate(parts).astype(np.float32)


# 2 bodies x (63 + 63 + 1 + 20 + 2) + match_frame + frames_next_turn
OBS_DIM = 2 * (63 + 63 + 1 + 20 + 2) + 2


class FrozenNetworkOpponent:
    """Runs a frozen torch module as an opponent (see the module docstring)."""

    def __init__(
        self,
        network,
        deterministic: bool = True,
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        import torch

        self._torch = torch
        self._device = torch.device(device)
        self._network = network.to(self._device).eval()
        for parameter in self._network.parameters():
            parameter.requires_grad_(False)
        self.deterministic = deterministic
        self._seed = seed
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(0 if seed is None else seed)

    @classmethod
    def from_torchscript(cls, path, **kwargs) -> FrozenNetworkOpponent:
        import torch

        return cls(torch.jit.load(str(path), map_location="cpu"), **kwargs)

    def reset(self, seed: int | None = None) -> None:
        if seed is None:
            seed = self._seed
        self._generator.manual_seed(0 if seed is None else seed)

    def act(self, obs: dict) -> np.ndarray:
        torch = self._torch
        x = torch.from_numpy(flatten_observation(obs))[None].to(self._device)
        with torch.no_grad():
            logits = self._network(x)
            logits = logits.reshape(NUM_CONTROLLABLES, NUM_JOINT_STATES).float().cpu()
            if self.deterministic:
                action = logits.argmax(dim=-1)
            else:
                probabilities = torch.softmax(logits, dim=-1)
                action = torch.multinomial(
                    probabilities, 1, generator=self._generator
                ).squeeze(-1)
        return action.numpy().astype(np.int64)
