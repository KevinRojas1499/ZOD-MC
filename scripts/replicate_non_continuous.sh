python3 experiments.py --config config/qualitative_comparisons.yaml \
    --mode eval_mmd --T 2 --disc_steps 25 \
    --sampling_method ei --ula_step_size 0.1 \
    --sampling_batch_size 1000 --num_batches 1 \
    --score_method p0t --density gmm --dimension 2 \
    --density_parameters_path config/density_parameters/2d_gmm.yaml \
    --discontinuity --eval_mmd \
    --save_folder plots/disc_gmm/
    