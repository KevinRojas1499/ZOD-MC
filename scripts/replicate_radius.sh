python3 experiments.py --config config/mmd_experiments.yaml \
    --mode radius --T 10 --disc_steps 50 --sampling_eps 5e-3 \
    --sampling_method ei \
    --sampling_batch_size 1000 --num_batches 1 \
    --score_method p0t --density gmm --dimension 2 \
    --proximal_M 20 --proximal_num_iters 5000 \
    --density_parameters_path config/density_parameters/2d_gmm.yaml \
    --save_folder plots/radius/ --samples_ckpt plots/radius/samples_gmm.pt \
    # --load_from_ckpt # Comment to not load it
