[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# spectromer
Code repository for the Spectromer project

# Setup

Ubuntu dependencies for running:

$ sudo apt install python3.10-venv 

$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt

If you would like to upload execution status to WanDB, then:
$ wandb login

# Obtaining data

# Running

## First generate an image directory to use as input to the model

$ ./spectra_dataset_builder.py --dataset=sdss-tiny --imagedir=png-sdss-tiny-highsnr --csvdir=csv-sdss-tiny-highsnr --debug --filterlowsnr

## To run fine-tuning of the data: 

$ ./reg_vit_spectra.py --do_train --save_model --experiment sdss-small-test --dataset=sdss-small --imagedir=png-sdss-small-highsnr --debug

## To run predictions from a model which already went through fine-tuning 

$ ./reg_vit_spectra.py --do_eval --load_model_dir ./output-wimbledon --imagedir=png-sdss-small-highsnr --dataset=sdss-small --output_predictions predictions.csv --debug

# Extra options

If you want to silence output to WanDB for a specific run, export WANDB_SILENT=true

In systems with more than one GPU, in case you want to train with both but they are asymmetrical, export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1

## Citation

If you use this software in your research or any publications, please cite our accompanying paper:

> **Applying Vision Transformers on Spectral Analysis of Astronomical Objects**  
> Luis Felipe Strano Moraes, Ignacio Becker, Pavlos Protopapas, Guillermo Cabrera-Vives  
> arXiv:2506.00294 [astro-ph.IM]  
> <https://arxiv.org/abs/2506.00294>

Or use the following BibTeX entry:

```bibtex
@article{strano2025vision,
  title        = {Applying Vision Transformers on Spectral Analysis of Astronomical Objects},
  author       = {Strano Moraes, Luis Felipe and Becker, Ignacio and Protopapas, Pavlos and Cabrera-Vives, Guillermo},
  journal      = {arXiv preprint arXiv:2506.00294},
  year         = {2025},
  eprint       = {2506.00294},
  archivePrefix= {arXiv},
  primaryClass = {astro-ph.IM}
}

