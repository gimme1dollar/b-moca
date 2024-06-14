import yaml
import argparse
from bmoca.agent.custom.rl.workspace import Workspace

from bmoca.utils import set_random_seed

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--config', help="configuration file *.yaml", 
                        type=str, required=False, default='./config/ppo.yaml')
    args = parser.parse_args()
    return args


def dict_to_namespace(d):
    namespace = argparse.Namespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict_to_namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace


if __name__ == "__main__":
    args = parse_args()
    set_random_seed(args.seed)

    opt = yaml.load(open(args.config), Loader=yaml.FullLoader) # dictionary
    args = vars(args)
    args.update(opt)
    args = dict_to_namespace(args)

    workspace = Workspace(args)
    try:
        workspace.run()
    except Exception as e:
        print("="*10, e)
        workspace.close()
    
print("all done")
