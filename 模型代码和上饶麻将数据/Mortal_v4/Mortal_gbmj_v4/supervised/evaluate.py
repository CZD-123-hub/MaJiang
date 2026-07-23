import argparse
import json
import logging
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from supervised.config import PROJECT_ROOT, config
from supervised.data_module import build_loader
from supervised.loops import evaluate
from supervised.model import Res2NetPolicyModel
from supervised.splits import build_data_splits


VALID_SPLITS = ('val', 'test')


def resolve_checkpoint_path(checkpoint_spec, control_cfg):
    if checkpoint_spec in ('best', 'latest'):
        ckpt_name = 'gbmj_policy_best.pth' if checkpoint_spec == 'best' else 'gbmj_policy_latest.pth'
        return PROJECT_ROOT / control_cfg['checkpoint_dir'] / ckpt_name
    checkpoint_path = Path(checkpoint_spec)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    return checkpoint_path


def run_evaluation(split=None, checkpoint=None, max_batches=None):
    control_cfg = config['control']
    dataset_cfg = config['dataset']
    eval_cfg = config.get('eval', {})
    split = split or str(eval_cfg.get('default_split', 'test'))
    checkpoint_spec = checkpoint or str(eval_cfg.get('checkpoint', 'best'))
    config_max_batches = int(eval_cfg.get('default_max_batches', 0) or 0)
    max_batches = max_batches if max_batches is not None else (None if config_max_batches <= 0 else config_max_batches)
    if split not in VALID_SPLITS:
        raise ValueError(f'unsupported split: {split}')

    device = torch.device(control_cfg['device'])
    enable_amp = bool(control_cfg.get('enable_amp', False))
    loss_cfg = config.get('loss', {})
    value_loss_weight = float(loss_cfg.get('value_loss_weight', 0.0))

    splits = build_data_splits(dataset_cfg)
    files = splits[split]
    if not files:
        raise RuntimeError(f'{split} split is empty')

    loader = build_loader(files, dataset_cfg, control_cfg, training=False)
    ckpt_path = resolve_checkpoint_path(checkpoint_spec, control_cfg)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    checkpoint_config = state.get('config', config)
    model = Res2NetPolicyModel(**checkpoint_config['model']).to(device)
    model.load_state_dict(state['model'])

    metrics = evaluate(
        model,
        loader,
        device,
        enable_amp,
        max_batches,
        value_loss_weight=value_loss_weight,
    )
    return {
        'split': split,
        'checkpoint_spec': checkpoint_spec,
        'checkpoint': str(Path(ckpt_path).resolve()),
        'checkpoint_steps': int(state.get('steps', -1)),
        'checkpoint_best_metrics': state.get('best_metrics'),
        'files': len(files),
        'metrics': metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=VALID_SPLITS, default=None)
    parser.add_argument('--checkpoint', default=None, help='best/latest or a checkpoint path')
    parser.add_argument('--max-batches', type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    result = run_evaluation(split=args.split, checkpoint=args.checkpoint, max_batches=args.max_batches)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
