import configargparse
import run_toy

def parse_arguments():
    p = configargparse.ArgParser(description='Arguments for nonconvex sampling')

    p.add('-c','--config', is_config_file=True)

    # Files
    p.add_argument('--work_dir')
    p.add_argument('--ckpt_dir')
    p.add_argument('--samples_dir')    
    
    # Checkpoint path
    p.add_argument('--ckpt_path',required=False)


    # Mode
    p.add_argument('--mode', choices=['train','sample'])

    # Optimizer
    p.add_argument('--optimizer',choices=['Adam'])
    p.add_argument('--lr',type=float)

    # Training
    p.add_argument('--train_iters',type=int)
    p.add_argument('--batch_size',type=int)
    p.add_argument('--snapshot_freq',type=int)

    #Ode Solver
    p.add_argument('--atol',type=float)
    p.add_argument('--rtol',type=float)
    p.add_argument('--t1',type=int)


    # Problem Specifics
    p.add_argument('--density',choices=['gmm','double-well'])
    p.add_argument('--coeffs',type=float, action='append')
    p.add_argument('--means', type=float, action='append')
    p.add_argument('--variances', type=float, action='append')

    return p.parse_args()
def main(config):
    print("")
    if config.density == 'gmm':
        if config.mode == 'train':
            run_toy.train(config)
        elif config.mode == 'sample':
            run_toy.eval(config)
        else:
            print("Mode doesn't exist")
    else:
        print("Dataset not implemented yet")
            
    


if __name__ == '__main__':
    config = parse_arguments()
    print(config)
    main(config)