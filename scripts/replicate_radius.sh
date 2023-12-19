python3 main.py --config config/general_config.yaml \
    --mode radius --T 10 --disc_steps 50 --sampling_eps 5e-3 \
    --sampling_method ei \
    --sampling_batch_size 1000 --num_batches 1 \
    --score_method p0t --density gmm --dimension 2