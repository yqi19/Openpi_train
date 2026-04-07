"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import argparse
import dataclasses

import numpy as np
import tqdm

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


@dataclasses.dataclass(frozen=True)
class DataRepoOverride:
    """Compatibility shim for passing --data.repo_id to this script.

    We only override the data `repo_id`, while keeping the config's original concrete DataConfigFactory
    (and therefore its transforms/repack/model transforms).
    """

    repo_id: str | None = None


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(config_name: str, max_frames: int | None = None, data: DataRepoOverride = DataRepoOverride()):
    config = _config.get_config(config_name)
    if data.repo_id is not None:
        # Keep the concrete data factory type (so transforms stay correct) but override repo_id.
        config = dataclasses.replace(config, data=dataclasses.replace(config.data, repo_id=data.repo_id))

    if max_frames is None:
        max_frames = config.compute_norm_stats_max_frames

    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    # Training-time loading uses `assets_dir / asset_id` (not `assets_dir / repo_id`).
    # For backward compatibility, we also keep saving under `assets_dir / repo_id`.
    # Note: when `repo_id` is an absolute path, `assets_dir / repo_id` becomes the absolute path itself.
    asset_out_dir = config.assets_dirs / data_config.asset_id if data_config.asset_id is not None else None
    repo_out_dir = config.assets_dirs / data_config.repo_id

    if asset_out_dir is not None and asset_out_dir != repo_out_dir:
        print(f"Writing stats to asset_id dir: {asset_out_dir}")
        normalize.save(asset_out_dir, norm_stats)

    print(f"Writing stats to repo_id dir: {repo_out_dir}")
    normalize.save(repo_out_dir, norm_stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute normalization stats for a given config.")
    parser.add_argument(
        "config_name",
        nargs="?",
        default=None,
        help="Config name to compute norm stats for (e.g. pi05_maniskill).",
    )
    parser.add_argument("--config-name", dest="config_name_opt", default=None, help="Alias for config name.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on number of frames.")
    # Compat: accept both underscore and hyphen variants.
    parser.add_argument("--data.repo_id", dest="data_repo_id", default=None)
    parser.add_argument("--data.repo-id", dest="data_repo_id", default=None)

    args = parser.parse_args()
    resolved_config_name = args.config_name_opt or args.config_name
    if not resolved_config_name:
        parser.error("Missing config name. Provide positional <config_name> or --config-name <config_name>.")

    main(
        resolved_config_name,
        max_frames=args.max_frames,
        data=DataRepoOverride(repo_id=args.data_repo_id),
    )
