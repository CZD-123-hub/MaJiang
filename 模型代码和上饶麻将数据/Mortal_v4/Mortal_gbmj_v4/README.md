# Mortal GBMJ V3

This project keeps the mature 235-way GBMJ action convention, but replaces the
paper-reproduction encoder/backbone with a practical v3 supervised model:

- visible-state encoder: `194 x 4 x 9`
- policy output: `235` classes
- data: public `../dataset/version3/` logs
- augmentation: `12x` suit permutation plus number mirror
- backbone: no-downsample multi-scale CNN + spatial Res2Net blocks
- neck: light tile-grid Transformer over the 36 `4 x 9` positions
- optimizer: `Adam`
- loss: cross entropy

## V3 Feature Planes

The encoder starts from the paper's visible features, but uses a longer discard
history and adds cheap state features already maintained by `PlayerState`.

- seat wind: `1`
- round wind: `1`
- public tiles already appeared: `4`
- hand tiles: `4`
- all players' chow/pung/kong meld summaries: `24`
- discard history: `4 x 28 = 112`
- visible/remaining/dead tile counts: `3`
- waits, shanten, discard candidates, keep/next-shanten hints: `12`
- current action context and candidates: `19`
- recent discard decay by player: `4`
- compact table scalar planes: `10`

Total: `194`

The encoder does not use hidden hands or true wall information.

## Model

`Res2NetPolicyModel` now builds `MahjongCNNTransformerBackbone`:

- parallel `1x3`, `1x5`, `3x1`, and `3x3` CNN stem
- Res2Net-style residual blocks without spatial downsampling
- squeeze-excitation channel attention
- 36 tile-grid tokens plus a CLS token through a small Transformer
- final 235-way policy head

## Train

```bash
python -m supervised.train
```

## Evaluate

```bash
python -m supervised.evaluate --checkpoint best --split val
```

## Export Inference Weights

```bash
python -m supervised.export_inference --checkpoint best
```
