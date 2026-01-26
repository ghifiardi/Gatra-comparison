from architecture_a_rl.env_bandit import compute_reward, RewardConfig

def test_reward_threat_good_action() -> None:
    cfg = RewardConfig(10, -3, -15, 1, {"escalate": 3, "contain": 2, "monitor": 1, "dismiss": 0.5})
    r = compute_reward("threat", 1.0, "escalate", cfg)
    assert r > 0


def test_reward_benign_bad_action() -> None:
    cfg = RewardConfig(10, -3, -15, 1, {"escalate": 3, "contain": 2, "monitor": 1, "dismiss": 0.5})
    r = compute_reward("benign", 0.0, "escalate", cfg)
    assert r < 0
