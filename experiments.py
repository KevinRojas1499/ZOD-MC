import configargparse
import run_toy
import estimator_accuracy_experiments
import generation_accuracy_experiments
import fourier_experiments
import experiments_based_on_p0t
import mmd_loss_comparisons
import radius_increase_experiments

def parse_arguments():
    p = configargparse.ArgParser(description='Arguments for nonconvex sampling')

    p.add('-c','--config', is_config_file=True)

    # Mode
    p.add_argument('--mode', choices=['eval_mmd','radius'])
    p.add_argument('--score_method', choices=['p0t','recursive'],default='p0t')
    p.add_argument('--p0t_method', choices=['rejection','ula'],default='rejection')
    p.add_argument('--dimension', type=int)
    
    # Experiments to run
    p.add_argument('--eval_mmd', action='store_true',default=False)
    p.add_argument('--methods_to_run',action='append', default=[])
    p.add_argument('--num_samples_for_rdmc',type=int)
    p.add_argument('--sampling_eps_rdmc', type=float) # early stopping
    p.add_argument('--sampling_eps_rejec', type=float) # early stopping
    
    p.add_argument('--min_num_iters_rdmc',type=int)
    p.add_argument('--max_num_iters_rdmc',type=int)
    p.add_argument('--iters_rdmc_step',type=int)
    
    # Baselines
    p.add_argument('--baselines',action='append', default=[])
    p.add_argument('--langevin_step_size',type=float)
    
    p.add_argument('--proximal_M',type=float)
    p.add_argument('--proximal_num_iters',type=int)
    
    # Sampler details
    p.add_argument('--max_iters_optimization',type=int, default=50)
    p.add_argument('--num_sampler_iterations', type=int) # For langevin
    p.add_argument('--ula_step_size',type=float)
    p.add_argument('--num_estimator_batches', type=int, default=1) # For rejection
    p.add_argument('--num_estimator_samples', type=int, default=10000) # Per batch for rejection
    p.add_argument('--gradient_estimator',choices=['conv','direct']) # For quotient estimator
    p.add_argument('--eps_stable',type=float, default=1e-9) # For quotient based methods
    p.add_argument('--num_recursive_steps',type=int, default=6)
    
    # SDE Parameters
    p.add_argument('--sde_type', choices=['vp'], default='vp')
    p.add_argument('--multiplier', default=0, type=float)
    p.add_argument('--bias', default=2., type=float)

    # Sampling Parameters
    p.add_argument('--sampling_method', choices=['ei','em'])
    p.add_argument('--num_batches', type=int)
    p.add_argument('--sampling_batch_size',type=int)
    p.add_argument('--T', type=float) # early stopping    
    p.add_argument('--sampling_eps', type=float) # early stopping
    p.add_argument('--disc_steps',type=int)
    p.add_argument('--ula_steps',type=int,default=0) # Finish off with some ula steps

    # Problem Specifics
    p.add_argument('--density',choices=['gmm','double-well','mueller','funnel'])
    p.add_argument('--density_parameters_path',type=str)

    return p.parse_args()

def main(config):
    if config.mode == 'eval_mmd':
        mmd_loss_comparisons.eval(config)
    elif config.mode == 'radius':
        radius_increase_experiments.eval(config)
    else:
        print("Mode hasn't been implemented")



if __name__ == '__main__':
    config = parse_arguments()
    print(config)
    main(config)