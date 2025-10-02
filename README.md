# Zeroth-Order Sampling Methods for Non-Log-Concave Distributions

This is the official repo for the ZOD-MC.



### Setting up:
To install the required packages simply run:

```bash
conda create --name zodmc python=3.12
pip install pip --upgrade
pip install -r requirements.txt
```

### Sampling
We use wandb to track our experiments, if you don't have a wandb account you can use the `sample.py` file and process the samples as you want. If you do have wandb, you can sample using a diffusion based sampler from a 2d gaussian mixture by running:

```bash
python3 main.py --config config/general_config.yaml \
    --density_parameters_path config/density_parameters/2d_gmm.yaml \
    --density gmm --dimension 2
```
#### Table of config hyperparmeters to pick a diffusion method
| Method        | score_method  | p0t_method |
| ------------- |:-------------:| ----------:|
| ZOD-MC        | p0t           | rejection  |
| RDMC          | p0t           |   ula      |
| RSDMC         | recursive     |    -       |

## Your own distributions

You can implement your own distributions by adding them in the `utils/densities.py` file. You can add a distribution from a potential by passing it to `DistributionFromPotential`, and the gradients will be computed using pytorchs automatic differentiation.


### Replicating our results:

After installing the required packages you may replicate all our results in the paper by running the following commands:

```bash
bash scripts/replicate_2d_mmd.sh
bash scripts/replicate_radius.sh
bash scripts/replicate_non_continuous.sh
bash scripts/replicate_mueller.sh
bash scripts/replicate_dimension.sh
bash scripts/replicate_score_error.sh
```

