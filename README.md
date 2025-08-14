# Investigating Demographic Attribute Representation in Vision Encoders (IDARVE)

### Setting up the Virtual Environment

```
conda env create -f environment.yaml
```

### Loading the Checkpoints

```
bash setup/download_checkpoints.sh
```

### Setting up Datasets

From the repo's directory, run

```
python -m src.setup_datasets.ve_latent_dataset
```

```
python -m src.setup_datasets.sae_latent_dataset
```

### Analyzing Data Using PatchSAE

From the repo's directory, run the patchSAE notebook

```
jupyter notebook datasets/patchSAE_style_analysis.ipynb
```

### (Optional) Train the SAE and Linear Probes

Before training, setup the model and trainer in `config/config.yaml`. To adjust hyperparameters, adjust the individual model and trainer configs, which can be found under `config/model` and `config/trainer`, respectively.

To train, run the following from the repo's directory:

```
python -m src.main
```
