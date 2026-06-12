<div align="center">
  <h1>Rethinking One-Step Image Editing through ChordEdit: Reproduction, Simplification, and New Insights</h1>
  <!-- <h2>[CVPR 2026 Oral] ChordEdit: One-Step Low-Energy Transport for Image Editing</h2> -->
  <div class="authors">
    <span>Minghan Li, Jeremy Moebel, and Mengyu Wang</span><br>
    <span style="color: #666; font-size: 0.9em;">Harvard AI and Robotics Lab, Harvard University</span><br>
    <span style="color: #777; font-size: 0.85em;">Reproduction and analysis of the original ChordEdit paper ([CVPR 2026 Oral]) by Liangsi Lu, Xuhang Chen, Minzhe Guo, Shichu Li, Jingchao Wang, and Yang Shi.</span>
  </div>

  <a href="https://chordedit.github.io"><img src="https://img.shields.io/badge/Project-Page-2b7de9"></a>
  <a href="https://arxiv.org/pdf/2602.19083"><img src="https://img.shields.io/badge/ChordEdit-arXiv-b31b1b.svg"></a>
  <a href=""><img src="https://img.shields.io/badge/Reproduction-arXiv_TBD-b31b1b.svg"></a>

  <img src="chord_show.gif" alt="ChordEdit demo" width="100%" />
</div>

## 0. Reproduced ChordEdit Results
This averages the 700 PIE-Bench samples in PIE-Bench.

| Method | Structure Dist. ↓ | PSNR ↑ | LPIPS ↓ | MSE ↓ | SSIM ↑ | CLIP Src. ↑ | CLIP Tgt. ↑ | CLIP Edit ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ChordEdit (Naive, paper)   |0.02544 | 21.38  | 0.1313 | 0.00973 | 0.7439 |  -      | 25.11   | 21.96.  |
| ChordEdit (Naive, Reprod.) |0.0457 | 20.14 | 0.1559 | 0.0133  | 0.7327 | 25.4281 | 25.1151 | 22.5767 |
| ChordEdit (Paper)   | 0.01658      | 22.20   | 0.12825 | 0.00684   | 0.7591      | - | 25.58   | 22.96 |
| ChordEdit (Reprod.)    | 0.0295 | 22.64   | 0.1185  | 0.0080 | 0.7675 | 25.4281 | 24.8226 | 22.1554 |

To reproduce the ablation results:

1. Download PIE-Bench from the official GitHub repository and place it at:
   ```text
   datasets/PIE-Bench_v1
   ```
2. Run the Stable Diffusion ablation script:
   ```bash
   bash scripts/run_sd_ablation.sh
   ```
3. Evaluate the generated outputs with the external evaluation script:
   ```bash
   bash external/evaluation/run_eval.sh
   ```
4. Plot and merge the ablation results:
   ```bash
   python scripts/plot_performance_ablation.py
   ```


## 1. Environment
- Python 3.12
- PyTorch 2.5.0
- This repository requires the `sd-turbo` weights: https://huggingface.co/stabilityai/sd-turbo
- Model root should contain:
  - `unet/`
  - `scheduler/`
  - `text_encoder/`
  - `tokenizer/`
  - `vae/`

## 2. Install Dependencies
```bash
pip install -r requirement.txt
```

## 3. Run the Web Demo
Launch the interactive demo:
```bash
python app.py --model-root /path/to/sd-turbo --server-port 7860
```

Running `python app.py` now launches a local Gradio web app.
- Left panel: upload the original image, set source prompt, target prompt, and tuning parameters.
- Right panel: view the edited output image.
- Bottom section: click built-in examples (image + source prompt + target prompt) to auto-fill inputs.

<img src="chord_app.png" alt="ChordEdit app" width="100%" />

## 4. Run PIE Benchmark Export
Run PIE-Bench export with:
```bash
python run_pie_bench.py --model-root /path/to/sd-turbo --pie-root /path/to/pie_bench
```
`--pie-root` should point to a PIE-Bench folder containing at least:

1. `annotation_images/` — original PIE-Bench images (subfolders keep the official naming).
2. `mapping_file.json` — the mapping metadata describing prompts, instructions, and masks.

Example layout:
```
pie_bench
|-annotation_images
|-mapping_file.json
```

For PIE-Bench data preparation and protocol details, please refer to:
https://github.com/cure-lab/PnPInversion


# Citation
If you find our work helpful, please **star 🌟** this repo and **cite 📑** our paper. Thanks for your support!
```
@article{lu2026chordedit,
  title={ChordEdit: One-Step Low-Energy Transport for Image Editing},
  author={Lu, Liangsi and Chen, Xuhang and Guo, Minzhe and Li, Shichu and Wang, Jingchao and Shi, Yang},
  journal={arXiv preprint arXiv:2602.19083},
  year={2026}
}
```
