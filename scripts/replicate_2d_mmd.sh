python3 experiments.py --config config/mmd_experiments.yaml \
    --mode eval_mmd --T 2 --disc_steps 25 \
    --sampling_method ei --ula_step_size 0.1 \
    --sampling_batch_size 1000 --num_batches 1 \
    --score_method p0t --density lmm --dimension 2 \
    --density_parameters_path config/density_parameters/2d_lmm.yaml 