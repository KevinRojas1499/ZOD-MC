import configargparse
import run_toy

def parse_arguments():
    p = configargparse.ArgParser(description='Arguments for nonconvex sampling')

    p.add('-c','--config', is_config_file=True)

    # Mode
    p.add_argument('--mode', choices=['train','sample'])

    # Optimizer
    p.add_argument('--optimizer',choices=['Adam'])
    p.add_argument('--lr',type=float)

    # Training
    p.add_argument('--train_iters',type=int)
    p.add_argument('--batch_size',type=int)

    #Ode Solver
    p.add_argument('--atol',type=float)
    p.add_argument('--rtol',type=float)


    # Problem Specifics
    p.add_argument('--density',choices=['gmm','double-well'])
    p.add_argument('--coeffs',type=float, action='append')
    p.add_argument('--means', type=float, action='append')
    p.add_argument('--variances', type=float, action='append')

    return p.parse_args()
def main(config):
    print("")
    if config.density == 'gmm':
        run_toy.train(config)


if __name__ == '__main__':
    config = parse_arguments()
    print(config)
    main(config)