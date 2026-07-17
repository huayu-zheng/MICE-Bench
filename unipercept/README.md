# UniPercept runtime for MICE-Bench

This directory contains the UniPercept inference source used by creation Q3.
The upstream project is available at
[thunderbolt215/UniPercept](https://github.com/thunderbolt215/UniPercept), and
the official checkpoint is hosted at
[Thunderbolt215215/UniPercept](https://huggingface.co/Thunderbolt215215/UniPercept).

Only runtime source is vendored here. Upstream documentation images, training
assets, datasets, and model weights are intentionally excluded from this Git
repository.

## Setup

From the MICE-Bench repository root:

```bash
conda create -n mice-unipercept python=3.10 -y
conda activate mice-unipercept
pip install -r unipercept/requirements-mice.txt
huggingface-cli download Thunderbolt215215/UniPercept \
  --local-dir unipercept/ckpt/UniPercept
```

FlashAttention is optional. Test the deployment with a repository image:

```bash
python unipercept/test_mice.py --no-flash-attn
```

See the [main README](../README.md#deploy-unipercept-for-creation-q3) for the
complete MICE-Bench Q3 workflow.

