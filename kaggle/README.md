# CMF 120M Kaggle Runbook

This folder is the Kaggle handoff for the CMF Infinity 120M-class model.

It does not claim AGI. It gives you a runnable path to continue pretraining or
fine-tuning a 120M-class CMF package on Kaggle hardware, then prompt the saved
package.

## Files

- `train_cmf_120m.py`: prepares or reuses a GPT-2 token cache and trains CMF.
- `infer_cmf_120m.py`: prompts a saved CMF `.package.pt`.
- `requirements.txt`: optional packages if the Kaggle image is missing them.

## Typical Kaggle Commands

From a Kaggle notebook cell, after uploading/unzipping this bundle:

```bash
pip install -q -r kaggle/requirements.txt
python kaggle/train_cmf_120m.py \
  --device cuda \
  --preset infinity-0.12b \
  --max-tokens 2000000 \
  --steps 1000 \
  --seq-len 128 \
  --micro-batch-size 1 \
  --grad-accum 16
```

If you uploaded a starting package as a Kaggle dataset:

```bash
python kaggle/train_cmf_120m.py \
  --device cuda \
  --init-package /kaggle/input/cmf-bundle/pretrained/cmf_120m_gate.package.pt \
  --max-tokens 2000000 \
  --steps 1000
```

Prompt the resulting package:

```bash
python kaggle/infer_cmf_120m.py \
  --package /kaggle/working/cmf_120m/checkpoints/cmf_120m_kaggle.package.pt \
  --prompt "Explain CMF in one short sentence:"
```

## Notes

- If Kaggle internet is disabled, upload a token cache and pass `--token-cache`.
- The default output package is `/kaggle/working/cmf_120m/checkpoints/cmf_120m_kaggle.package.pt`.
- More tokens matter more than clever naming. A serious 120M run needs many
  millions to billions of tokens plus held-out evals.
