import json
import os
from pathlib import Path

import torch
import fire

from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from alphagen.data.expression import *
from alphagen.data.parser import ExpressionParser
from alphagen.models.linear_alpha_pool import LinearAlphaPool, MseAlphaPool
from alphagen.rl.env.wrapper import AlphaEnv
from alphagen.rl.policy import LSTMSharedNet
from alphagen.utils import reseed_everything, get_logger
from alphagen.rl.env.core import AlphaEnvCore
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen_qlib.stock_data import initialize_qlib, StockData


def build_parser() -> ExpressionParser:
    return ExpressionParser(
        Operators,
        ignore_case=True,
        non_positive_time_deltas_allowed=False,
        additional_operator_mapping={
            "Max": [Greater],
            "Min": [Less],
            "Delta": [Sub]
        }
    )


class CustomCallback(BaseCallback):
    def __init__(self, save_path, test_calculators, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.test_calculators = test_calculators
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self.logger.record('pool/size', self.pool.size)
        self.logger.record('pool/significant', (abs(self.pool.weights[:self.pool.size]) > 1e-4).sum())
        self.logger.record('pool/best_ic_ret', self.pool.best_ic_ret)
        self.logger.record('pool/eval_cnt', self.pool.eval_cnt)
        n_days = sum(c.data.n_days for c in self.test_calculators)
        ic_test_mean, rank_ic_test_mean = 0., 0.
        for i, test_calculator in enumerate(self.test_calculators, start=1):
            ic_test, rank_ic_test = self.pool.test_ensemble(test_calculator)
            ic_test_mean += ic_test * test_calculator.data.n_days / n_days
            rank_ic_test_mean += rank_ic_test * test_calculator.data.n_days / n_days
            self.logger.record(f'test/ic_{i}', ic_test)
            self.logger.record(f'test/rank_ic_{i}', rank_ic_test)
        self.logger.record('test/ic_mean', ic_test_mean)
        self.logger.record('test/rank_ic_mean', rank_ic_test_mean)
        self.save_checkpoint()

    def save_checkpoint(self):
        path = os.path.join(self.save_path, f'{self.num_timesteps}_steps')
        self.model.save(path)
        if self.verbose > 1:
            print(f'Saving model checkpoint to {path}')
        with open(f'{path}_pool.json', 'w') as f:
            json.dump(self.pool.to_json_dict(), f)

    @property
    def pool(self) -> LinearAlphaPool:
        assert isinstance(self.env_core.pool, LinearAlphaPool)
        return self.env_core.pool

    @property
    def env_core(self) -> AlphaEnvCore:
        return self.training_env.envs[0].unwrapped  # type: ignore


def resume(
    checkpoint_dir: str,
    checkpoint_step: int,
    target_steps: int = 200_000,
    seed: int = 0,
    instruments: str = "csi300",
    pool_capacity: int = 10,
):
    """Resume a previously interrupted AlphaGen RL run from a saved checkpoint.

    :param checkpoint_dir: directory containing `{step}_steps.zip` / `{step}_steps_pool.json`
    :param checkpoint_step: which checkpoint step to resume from (e.g. 32768)
    :param target_steps: total step budget for the whole experiment (same meaning as
        the original `--steps` argument), training continues until this is reached
    :param seed: must match the original run's seed
    :param instruments: must match the original run's instruments
    :param pool_capacity: must match the original run's pool_capacity
    """
    reseed_everything(seed)
    initialize_qlib("~/.qlib/qlib_data/cn_data")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -20) / close - 1

    def get_dataset(start: str, end: str) -> StockData:
        return StockData(instrument=instruments, start_time=start, end_time=end, device=device)

    segments = [
        ("2012-01-01", "2021-12-31"),
        ("2022-01-01", "2022-06-30"),
        ("2022-07-01", "2022-12-31"),
        ("2023-01-01", "2023-06-30"),
    ]
    datasets = [get_dataset(*s) for s in segments]
    calculators = [QLibStockDataCalculator(d, target) for d in datasets]

    pool = MseAlphaPool(
        capacity=pool_capacity,
        calculator=calculators[0],
        ic_lower_bound=None,
        l1_alpha=5e-3,
        device=device,
    )

    pool_json_path = Path(checkpoint_dir) / f"{checkpoint_step}_steps_pool.json"
    with open(pool_json_path) as f:
        pool_state = json.load(f)
    parser = build_parser()
    exprs = [parser.parse(e) for e in pool_state["exprs"]]
    weights = pool_state["weights"]
    pool.force_load_exprs(exprs, weights)
    print(f"[Resume] Loaded {pool.size} alphas from {pool_json_path}, best_ic_ret={pool.best_ic_ret:.4f}")

    env = AlphaEnv(pool=pool, device=device, print_expr=True)

    model_zip_path = Path(checkpoint_dir) / f"{checkpoint_step}_steps.zip"
    model = MaskablePPO.load(str(model_zip_path), env=env)
    print(f"[Resume] Loaded model from {model_zip_path}, num_timesteps={model.num_timesteps}")

    remaining_steps = target_steps - model.num_timesteps
    if remaining_steps <= 0:
        print(f"[Resume] Already at or past target_steps={target_steps}, nothing to do.")
        return

    checkpoint_callback = CustomCallback(
        save_path=checkpoint_dir,
        test_calculators=calculators[1:],
        verbose=1,
    )

    print(f"[Resume] Continuing training for {remaining_steps} more steps "
          f"(from {model.num_timesteps} to {target_steps})")
    model.learn(
        total_timesteps=remaining_steps,
        callback=checkpoint_callback,
        reset_num_timesteps=False,
        tb_log_name=f"{Path(checkpoint_dir).name}_resumed",
    )


if __name__ == '__main__':
    fire.Fire(resume)
