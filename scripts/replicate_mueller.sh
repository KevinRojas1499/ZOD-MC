python3 main.py --config config/general_config.yaml \
    --mode eval_mmd --score_method p0t \
    --sampling_method ei --ula_step_size 0.0001 \
    --sampling_batch_size 1000 --num_batches 1 \
    --T 2 --disc_steps 25 \
    --density mueller --dimension 2 \
    --density_parameters_path config/density_parameters/mueller.yaml