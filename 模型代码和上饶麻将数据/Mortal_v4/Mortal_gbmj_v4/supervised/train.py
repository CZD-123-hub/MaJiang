import argparse
import logging
import sys
import time
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import optim
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from mortal_part.consts import obs_shape
from supervised.config import PROJECT_ROOT, config
from supervised.data_module import build_loader
from supervised.loops import RunningAverages, compute_policy_loss, evaluate
from supervised.model import Res2NetPolicyModel
from supervised.splits import build_data_splits


"""监督训练入口（建议按此文件顺序阅读）。

run_training 的执行链路是：读取配置 -> 划分文件 -> 构建 DataLoader ->
实例化模型 -> 前向计算交叉熵 -> AMP 反向传播与梯度裁剪 -> 定期验证、
记录 TensorBoard、保存 latest/best checkpoint。
"""


def parameter_count(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def save_checkpoint(path, model, optimizer, steps, epoch, best_metrics):
    # 完整 checkpoint 同时保存模型和优化器，适合中断后继续训练。
    torch.save(
        {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'steps': steps,
            'epoch': epoch,
            'best_metrics': best_metrics,
            'config': config,
        },
        path,
    )


def _inference_state_dict(model, fp16=False):
    state_dict = model.state_dict()
    if not fp16:
        return state_dict
    converted = {}
    for key, value in state_dict.items():
        if torch.is_floating_point(value):
            converted[key] = value.detach().to(dtype=torch.float16, device='cpu')
        else:
            converted[key] = value.detach().to(device='cpu')
    return converted


def save_inference_checkpoint(path, model, steps, epoch, best_metrics, fp16=False):
    # 推理 checkpoint 只保留模型权重和元信息，体积更小，供策略/PPO 加载。
    torch.save(
        {
            'model': _inference_state_dict(model, fp16=fp16),
            'steps': steps,
            'epoch': epoch,
            'best_metrics': best_metrics,
            'config': config,
            'inference_only': True,
            'dtype': 'float16' if fp16 else 'float32',
        },
        path,
    )


def write_metrics(writer, prefix, metrics, step):
    for key, value in metrics.items():
        if key in ('count', 'batches'):
            continue
        writer.add_scalar(f'{prefix}/{key}', float(value), step)


def is_better_checkpoint(metrics, best_metrics, selection_cfg):
    primary = str(selection_cfg.get('metric', 'action_acc'))
    secondary = str(selection_cfg.get('secondary_metric', 'discard_acc'))
    if best_metrics is None:
        return True
    if float(metrics[primary]) > float(best_metrics[primary]):
        return True
    if float(metrics[primary]) == float(best_metrics[primary]) and float(metrics[secondary]) > float(best_metrics[secondary]):
        return True
    if float(metrics[primary]) == float(best_metrics[primary]) and float(metrics[secondary]) == float(best_metrics[secondary]) and float(metrics['loss']) < float(best_metrics['loss']):
        return True
    return False


def run_training():
    control_cfg = config['control']
    dataset_cfg = config['dataset']
    model_cfg = config['model']
    optim_cfg = config['optim']
    loss_cfg = config.get('loss', {})
    selection_cfg = config['selection']

    # device 决定计算设备；开始正式训练前应确认 cuda 可用且显存足够。
    device = torch.device(control_cfg['device'])
    torch.backends.cudnn.benchmark = bool(control_cfg.get('enable_cudnn_benchmark', False))
    enable_amp = bool(control_cfg.get('enable_amp', False))

    # 先按“文件/对局”划分，再由 GameplayLoader 展开为状态样本，避免同局泄漏。
    splits = build_data_splits(dataset_cfg)
    for split_name, files in splits.items():
        logging.info('%s files: %s', split_name, f'{len(files):,}')
    if not splits['train']:
        raise RuntimeError('train split is empty')
    if not splits['val']:
        raise RuntimeError('val split is empty; set val_ratio > 0 or provide val_globs')

    train_loader = build_loader(splits['train'], dataset_cfg, control_cfg, training=True)
    val_loader = build_loader(splits['val'], dataset_cfg, control_cfg, training=False)

    # 模型输入通道必须和 obs_repr.py 的编码顺序/数量一致。
    model = Res2NetPolicyModel(**model_cfg).to(device)
    logging.info('obs shape=%s model params=%s', obs_shape, f'{parameter_count(model):,}')

    optimizer_name = str(optim_cfg.get('optimizer', 'adam')).lower()
    optimizer_cls = optim.AdamW if optimizer_name == 'adamw' else optim.Adam
    optimizer = optimizer_cls(
        model.parameters(),
        lr=float(optim_cfg['lr']),
        betas=tuple(optim_cfg['betas']),
        eps=float(optim_cfg['eps']),
        weight_decay=float(optim_cfg['weight_decay']),
    )
    # AMP 用半精度计算减少显存和提高吞吐；GradScaler 防止小梯度下溢。
    scaler = GradScaler(enabled=enable_amp)
    label_smoothing = float(loss_cfg.get('label_smoothing', 0.0))
    value_loss_weight = float(loss_cfg.get('value_loss_weight', 0.0))
    logging.info(
        'optimizer=%s lr=%s weight_decay=%s label_smoothing=%s value_loss_weight=%s',
        optimizer_name,
        optim_cfg['lr'],
        optim_cfg['weight_decay'],
        label_smoothing,
        value_loss_weight,
    )

    log_dir = PROJECT_ROOT / control_cfg['tensorboard_dir']
    ckpt_dir = PROJECT_ROOT / control_cfg['checkpoint_dir']
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))

    latest_file = ckpt_dir / 'gbmj_policy_latest.pth'
    best_file = ckpt_dir / 'gbmj_policy_best.pth'
    latest_infer_file = ckpt_dir / 'gbmj_policy_latest_infer.pth'
    latest_infer_fp16_file = ckpt_dir / 'gbmj_policy_latest_infer_fp16.pth'
    best_infer_file = ckpt_dir / 'gbmj_policy_best_infer.pth'
    best_infer_fp16_file = ckpt_dir / 'gbmj_policy_best_infer_fp16.pth'
    steps = 0
    start_epoch = 0
    best_metrics = None
    if latest_file.exists() and control_cfg.get('resume', True):
        state = torch.load(latest_file, map_location=device, weights_only=False)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        steps = int(state.get('steps', 0))
        start_epoch = int(state.get('epoch', 0))
        best_metrics = state.get('best_metrics')
        logging.info('resumed from %s at step %s epoch %s', latest_file, steps, start_epoch)

    log_every = int(control_cfg['log_every'])
    save_every = int(control_cfg.get('save_every', 0) or 0)
    eval_every = int(control_cfg.get('eval_every', 0) or 0)
    raw_eval_max_batches = int(control_cfg.get('eval_max_batches', 0) or 0)
    eval_max_batches = None if raw_eval_max_batches <= 0 else raw_eval_max_batches
    max_steps = int(control_cfg.get('max_steps', 0) or 0)
    max_epochs = int(control_cfg['max_epochs'])
    steps_per_epoch_estimate = int(control_cfg.get('steps_per_epoch_estimate', 0) or 0)
    max_grad_norm = float(optim_cfg['max_grad_norm'])
    running = RunningAverages()

    if max_steps > 0:
        pbar_total = max_steps
        pbar_initial = steps
        pbar_desc = 'TRAIN'
        use_epoch_progress = False
    elif steps_per_epoch_estimate > 0:
        pbar_total = max_epochs * steps_per_epoch_estimate
        pbar_initial = steps
        pbar_desc = 'TRAIN*'
        use_epoch_progress = False
    else:
        pbar_total = max_epochs
        pbar_initial = start_epoch
        pbar_desc = 'EPOCH'
        use_epoch_progress = True

    pbar = tqdm(total=pbar_total, initial=pbar_initial, dynamic_ncols=True, desc=pbar_desc)
    stop = False
    for epoch in range(start_epoch, max_epochs):
        epoch_start_time = time.time()
        epoch_steps = 0
        model.train()
        for batch in train_loader:
            # compute_policy_loss 内部完成数据搬运、前向、loss 和指标统计。
            loss, metrics = compute_policy_loss(
                model,
                batch,
                device,
                enable_amp,
                label_smoothing=label_smoothing,
                value_loss_weight=value_loss_weight,
            )
            if not torch.isfinite(loss.detach()):
                # [V4 metric/loss guard] Detach before formatting the failing
                # scalar; otherwise PyTorch emits an extra warning and hides the
                # useful component-loss context.
                denom = max(1, int(metrics.get('count', 1)))
                component_msg = (
                    f"policy={metrics.get('policy_loss_sum', 0.0) / denom:.6g}, "
                    f"value={metrics.get('value_loss_sum', 0.0) / denom:.6g}"
                )
                loss_value = float(loss.detach().float().cpu())
                raise RuntimeError(
                    f'non-finite loss at step {steps + 1}: {loss_value}; {component_msg}'
                )

            # 标准训练三步：缩放 loss 反传 -> 梯度裁剪 -> optimizer 更新。
            scaler.scale(loss).backward()
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                grad_norm = clip_grad_norm_(model.parameters(), max_grad_norm)
                if not torch.isfinite(grad_norm):
                    # [V4 metric/loss guard] Do not apply an optimizer step when
                    # AMP produced non-finite gradients.  This keeps the last
                    # good checkpoint usable instead of poisoning the weights.
                    logging.warning(
                        'skip non-finite gradient at step %s: grad_norm=%s',
                        steps + 1,
                        float(grad_norm.detach().float().cpu()),
                    )
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    continue
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            steps += 1
            epoch_steps += 1
            running.update(metrics)
            if use_epoch_progress:
                if steps % log_every == 0:
                    pbar.set_postfix(epoch=f'{epoch + 1}/{max_epochs}', step=steps, epoch_step=epoch_steps)
            else:
                pbar.update(1)
                if steps % log_every == 0:
                    pbar.set_postfix(epoch=f'{epoch + 1}/{max_epochs}', step=steps)

            if steps % log_every == 0:
                train_metrics = running.pop()
                write_metrics(writer, 'train', train_metrics, steps)
                writer.flush()

            if eval_every > 0 and steps % eval_every == 0:
                # 验证阶段不更新参数，只比较 loss/准确率并决定是否刷新 best。
                val_metrics = evaluate(
                    model,
                    val_loader,
                    device,
                    enable_amp,
                    eval_max_batches,
                    value_loss_weight,
                )
                write_metrics(writer, 'val', val_metrics, steps)
                writer.flush()
                logging.info(
                    'step=%s val_batches=%s val_loss=%.6f action_acc=%.4f discard_acc=%.4f meld_acc=%.4f',
                    steps,
                    val_metrics['batches'],
                    float(val_metrics['loss']),
                    float(val_metrics['action_acc']),
                    float(val_metrics['discard_acc']),
                    float(val_metrics['meld_acc']),
                )
                if is_better_checkpoint(val_metrics, best_metrics, selection_cfg):
                    best_metrics = dict(val_metrics)
                    save_checkpoint(best_file, model, optimizer, steps, epoch + 1, best_metrics)
                    save_inference_checkpoint(best_infer_file, model, steps, epoch + 1, best_metrics, fp16=False)
                    save_inference_checkpoint(best_infer_fp16_file, model, steps, epoch + 1, best_metrics, fp16=True)
                    logging.info('saved best checkpoint: %s', best_file)
                model.train()

            if save_every > 0 and steps % save_every == 0:
                save_checkpoint(latest_file, model, optimizer, steps, epoch + 1, best_metrics)
                save_inference_checkpoint(latest_infer_file, model, steps, epoch + 1, best_metrics, fp16=False)
                save_inference_checkpoint(latest_infer_fp16_file, model, steps, epoch + 1, best_metrics, fp16=True)
                logging.info('saved latest checkpoint: %s', latest_file)

            if max_steps > 0 and steps >= max_steps:
                stop = True
                break

        if running.batches > 0:
            train_metrics = running.pop()
            write_metrics(writer, 'train', train_metrics, steps)

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            enable_amp,
            eval_max_batches,
            value_loss_weight,
        )
        write_metrics(writer, 'val', val_metrics, steps)
        writer.flush()
        epoch_elapsed = time.time() - epoch_start_time
        if use_epoch_progress:
            pbar.update(1)
        pbar.set_postfix(
            epoch=f'{epoch + 1}/{max_epochs}',
            step=steps,
            epoch_step=epoch_steps,
            epoch_min=f'{epoch_elapsed / 60.0:.2f}',
        )
        logging.info(
            'epoch=%s/%s step=%s epoch_steps=%s epoch_minutes=%.2f val_batches=%s val_loss=%.6f action_acc=%.4f discard_acc=%.4f meld_acc=%.4f',
            epoch + 1,
            max_epochs,
            steps,
            epoch_steps,
            epoch_elapsed / 60.0,
            val_metrics['batches'],
            float(val_metrics['loss']),
            float(val_metrics['action_acc']),
            float(val_metrics['discard_acc']),
            float(val_metrics['meld_acc']),
        )
        if is_better_checkpoint(val_metrics, best_metrics, selection_cfg):
            best_metrics = dict(val_metrics)
            save_checkpoint(best_file, model, optimizer, steps, epoch + 1, best_metrics)
            save_inference_checkpoint(best_infer_file, model, steps, epoch + 1, best_metrics, fp16=False)
            save_inference_checkpoint(best_infer_fp16_file, model, steps, epoch + 1, best_metrics, fp16=True)
            logging.info('saved best checkpoint: %s', best_file)
        save_checkpoint(latest_file, model, optimizer, steps, epoch + 1, best_metrics)
        save_inference_checkpoint(latest_infer_file, model, steps, epoch + 1, best_metrics, fp16=False)
        save_inference_checkpoint(latest_infer_fp16_file, model, steps, epoch + 1, best_metrics, fp16=True)
        logging.info('saved latest checkpoint: %s', latest_file)

        if stop:
            break

    writer.close()
    pbar.close()
    return {
        'steps': steps,
        'best_metrics': best_metrics,
        'latest_checkpoint': str(latest_file),
        'best_checkpoint': str(best_file),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    run_training()


if __name__ == '__main__':
    main()
