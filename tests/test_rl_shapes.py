import torch
from architecture_a_rl.networks import Actor, Critic

def test_actor_critic_shapes() -> None:
    actor = Actor(state_dim=128, hidden=[64], action_dim=4)
    critic = Critic(state_dim=128, hidden=[64])

    x = torch.zeros((2, 128))
    probs = actor(x)
    values = critic(x)

    assert probs.shape == (2, 4)
    assert values.shape == (2,)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-6)
