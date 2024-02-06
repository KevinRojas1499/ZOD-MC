# Zeroth-Order Sampling Methods for Non-Log-Concave Distributions

This is the official repo for the ZOD-MC.



### Setting up:
To install the required packages simply run:

```{bash}
conda create --name zodmc
pip install pip --upgrade
pip install -r requirements.txt
```

### Sampling
You can install 

### Replicating our results:

After installing the required packages you may replicate all our results in the paper by running the following commands:

```{bash}
bash scripts/replicate_2d_mmd.sh
bash scripts/replicate_radius.sh
bash scripts/replicate_non_continuous.sh
bash scripts/replicate_mueller.sh
```

