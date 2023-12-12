import configargparse
import run_toy
import estimator_accuracy_experiments
import generation_accuracy_experiments
import fourier_experiments
import experiments_based_on_p0t

def parse_arguments():
    p = configargparse.ArgParser(description='Arguments for nonconvex sampling')

    p.add('-c','--config', is_config_file=True)

    # Wandb
    p.add_argument('--wandb_project_name', type=str)

    
    # Checkpoint path
    p.add_argument('--ckpt_path',required=False)


    # Mode
    p.add_argument('--mode', choices=['train','sample','estimator-experiments', 'generation-experiments','fourier-experiments','p0t-experiments'])
    p.add_argument('--score_method', choices=['convolution','quotient-estimator','trained','fourier', 'p0t'])
    p.add_argument('--p0t_method', choices=['proximal','rejection','random_walk','ula'])
    p.add_argument('--dimension', type=int)
    
    # Sampler details
    p.add_argument('--proximal_M', type=float)
    p.add_argument('--num_sampler_iterations', type=int)
    
    # Integrator details
    p.add_argument('--convolution_integrator', choices=['trap','simpson','mc'])
    p.add_argument('--integration_range', type=float)
    p.add_argument('--sub_intervals_per_dim',type=int)
   
    # Estimator information
    p.add_argument('--num_estimator_batches', type=int, default=1)
    p.add_argument('--num_estimator_samples', type=int, default=10000)
    p.add_argument('--eps_stable',type=float, default=1e-9)
    p.add_argument('--gradient_estimator',choices=['conv','direct'])

    # ODE Solver
    p.add_argument('--atol',type=float)
    p.add_argument('--rtol',type=float)
    p.add_argument('--t1',type=int)

    # SDE Parameters
    p.add_argument('--sde_type', choices=['vp','ve','edm'])
    p.add_argument('--sigma_min', type=float) # For VE
    p.add_argument('--sigma_max', type=float) # For VE
    p.add_argument('--multiplier', default=4, type=float)
    p.add_argument('--bias', default=0., type=float)


    # Sampling Parameters
    p.add_argument('--sampling_method', type=str)
    p.add_argument('--sampling_batch_size',type=int)
    p.add_argument('--num_batches', type=int)
    p.add_argument('--sampling_eps', type=float)
    p.add_argument('--disc_steps',type=int)
    p.add_argument('--ula_steps',type=int,default=0)

    # Problem Specifics
    p.add_argument('--density',choices=['gmm','double-well','mueller','funnel'])
    p.add_argument('--density_parameters_path',type=str)

    return p.parse_args()

def main(config):
    if config.mode == 'train':
        run_toy.train(config)
    elif config.mode == 'sample':
        run_toy.eval(config)
    elif config.mode == 'estimator-experiments':
        estimator_accuracy_experiments.run_experiments(config)
    elif config.mode == 'generation-experiments':
        generation_accuracy_experiments.run_experiments(config)
    elif config.mode == 'fourier-experiments':
        fourier_experiments.run_fourier_experiments(config)
    elif config.mode == 'p0t-experiments':
        experiments_based_on_p0t.run_experiments(config)
    else:
        print("Mode doesn't exist")



if __name__ == '__main__':
    config = parse_arguments()
    print(config)
    main(config)