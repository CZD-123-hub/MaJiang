# GBMJ Foresight C++ Extension

This module accelerates the 11-plane v4 foresight feature block:

- 7 route-family closeness planes
- 4 D=1 local-best discard planes

It compiles the copied `calsht` evaluator from `shanten-number-gbmj` and uses
the compact `shanten_shu.bin` / `shanten_zi.bin` lookup tables plus GBMJ route
metadata expected under `../data`.

The Python encoder sends concealed hand counts, fixed/open melds, and all
discard candidates in one call.  C++ then evaluates the current route closeness
and all candidate discards in one batch.

## Build

From the project root:

```bash
cd /data/mortal_gbmj/Mortal_gbmj_v4/foresight_cpp
python setup.py build_ext --inplace
```

Before training, put these files under `/data/mortal_gbmj/Mortal_gbmj_v4/data`:

```text
shanten_shu.bin
shanten_zi.bin
gbmj_*.bin
```

After build, this directory should contain a file named like:

```text
gbmj_foresight_cpp.cpython-311-x86_64-linux-gnu.so
```

`mortal_part/state/foresight.py` automatically imports this extension when it
exists. If it is not built, training falls back to the slower Python
implementation.
