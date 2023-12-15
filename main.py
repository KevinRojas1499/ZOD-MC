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
    p.add_argument('--tags', type=str)
    
    # Mode
    p.add_argument('--mode', choices=['sample','estimator-experiments', 'generation-experiments','fourier-experiments','p0t-experiments'])
    p.add_argument('--score_method', choices=['convolution','quotient-estimator','fourier', 'p0t'])
    p.add_argument('--p0t_method', choices=['rejection','ula'])
    p.add_argument('--dimension', type=int)
    
    # Sampler details
    p.add_argument('--max_iters_optimization',type=int, default=50)
    p.add_argument('--num_sampler_iterations', type=int) # For langevin
    p.add_argument('--ula_step_size',type=float)
    p.add_argument('--num_estimator_batches', type=int, default=1) # For rejection
    p.add_argument('--num_estimator_samples', type=int, default=10000) # Per batch for rejection
    p.add_argument('--gradient_estimator',choices=['conv','direct']) # For quotient estimator
    p.add_argument('--eps_stable',type=float, default=1e-9) # For quotient based methods
  
    # Integrator details for convolution method
    p.add_argument('--convolution_integrator', choices=['trap','simpson','mc'])
    p.add_argument('--integration_range', type=float)
    p.add_argument('--sub_intervals_per_dim',type=int)
   
    # SDE Parameters
    p.add_argument('--sde_type', choices=['vp'])
    p.add_argument('--multiplier', default=4, type=float)
    p.add_argument('--bias', default=0., type=float)

    # Sampling Parameters
    p.add_argument('--sampling_method', choices=['ei','em'])
    p.add_argument('--num_batches', type=int)
    p.add_argument('--sampling_batch_size',type=int)
    p.add_argument('--sampling_eps', type=float) # early stopping
    p.add_argument('--disc_steps',type=int)
    p.add_argument('--ula_steps',type=int,default=0) # Finish off with some ula steps

    # Problem Specifics
    p.add_argument('--density',choices=['gmm','double-well','mueller','funnel'])
    p.add_argument('--density_parameters_path',type=str)

    return p.parse_args()

def main(config):
    if config.mode == 'sample':
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