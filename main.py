import configargparse
import run_toy
import experiments

def parse_arguments():
    p = configargparse.ArgParser(description='Arguments for nonconvex sampling')

    p.add('-c','--config', is_config_file=True)

    # Wandb
    p.add_argument('--wandb_project_name', type=str)

    # Files
    p.add_argument('--work_dir')
    p.add_argument('--ckpt_dir')
    p.add_argument('--samples_dir')    

    p.add_argument('--num_eval_samples',type=int)
    
    # Checkpoint path
    p.add_argument('--ckpt_path',required=False)


    # Mode
    p.add_argument('--mode', choices=['train','sample','experiment'])
    p.add_argument('--score_method', choices=['convolution','quotient-estimator','trained'])
    p.add_argument('--dimension', type=int)
    # Integrator details
    p.add_argument('--convolution_integrator', choices=['trap','simpson','mc'])
    p.add_argument('--integration_range', type=float)
   
    # Estimator information
    p.add_argument('--num_estimator_samples', type=int, default=10000)
    p.add_argument('--eps_stable',type=float, default=1e-9)
    p.add_argument('--gradient_estimator',choices=['conv','direct'])

    # Optimizer
    p.add_argument('--optimizer',choices=['Adam'])
    p.add_argument('--lr',type=float)

    # Training
    p.add_argument('--train_iters',type=int)
    p.add_argument('--batch_size',type=int)
    p.add_argument('--snapshot_freq',type=int)

    # ODE Solver
    p.add_argument('--atol',type=float)
    p.add_argument('--rtol',type=float)
    p.add_argument('--t1',type=int)

    # SDE parameters
    p.add_argument('--sigma', type=float)

    # Sampling Parameters
    p.add_argument('--sampling_method', type=str)
    p.add_argument('--num_samples',type=int)
    p.add_argument('--sampling_eps', type=float)
    p.add_argument('--disc_steps',type=int)


    # Problem Specifics
    p.add_argument('--density',choices=['gmm','double-well'])
    p.add_argument('--density_parameters_path',type=str)

    return p.parse_args()
def main(config):
    if config.mode == 'train':
        run_toy.train(config)
    elif config.mode == 'sample':
        run_toy.eval(config)
    elif config.mode == 'experiment':
        experiments.run_experiments(config)
    else:
        print("Mode doesn't exist")



if __name__ == '__main__':
    config = parse_arguments()
    print(config)
    main(config)