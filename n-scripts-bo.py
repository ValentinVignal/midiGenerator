"""
Because of memory leak with tensorflow, I can't run a long bayesian optimization
This script is a solution to this issue. It run several time the bayesian-opt.py script and each time
it is loading the result of the previous one at the beginning
"""
import os
from epicpath import EPath
from termcolor import colored, cprint

from src import Args
from src.Args import Parser, ArgType

os.system('echo start n-script-bo.py')


def main(args):
    # To have colors from the first script
    # ----------------------------------------
    # Check that all the folder are available
    # ----------------------------------------
    if not args.in_place:
        if args.from_checkpoint is not None:
            if len(args.from_checkpoint.split('-')) == 1:
                name = 'bayesian_opt'
            else:
                name = f'bayesian_opt_{"_".join([str(s) for s in args.from_checkpoint.split("-")[:-1]])}'
            i_start = int(str(args.from_checkpoint).split('-')[-1]) + 1
        else:
            i_start = 0
            print('bo name', args.bo_name)
            name = 'bayesian_opt' if args.bo_name is None else f'bayesian_opt_{args.bo_name}'
        cprint('This script will need the next paths to be available:', 'red')
        for i in range(i_start, i_start + args.nscripts):
            path = EPath('hp_search', name + f'_{i}')
            cprint('path ' + path.as_posix(), 'yellow')
            if path.exists():
                raise FileExistsError(f'The folder "{path}" already exists')

    # ----------------------------------------
    # For the first Bayesian Optimization run, we just copy all the arguments except n-script
    # ----------------------------------------
    print(f'Script 1/{args.nscripts}')
    s = 'python bayesian-opt.py'
    for k, value in vars(args).items():
        if k not in ['nscripts', 'from_checkpoint', 'bo_name']:
            if k not in ['in_place', 'no_eager', 'pc', 'no_pc_arg', 'debug', 'seq2np', 'fast_seq', 'memory_seq', 'mono',
                         'from_checkpoint']:
                s += f' --{k.replace("_", "-")} {value}'
            elif value:
                s += f' --{k.replace("_", "-")}'
    if args.from_checkpoint is not None:
        s += f' --from-checkpoint {args.from_checkpoint}'
    if args.bo_name is not None:
        s += f' --bo-name {args.bo_name}'
    os.system(s)

    # ----------------------------------------
    # For all the others
    # ----------------------------------------
    if args.from_checkpoint:
        cp_id = args.from_checkpoint.split('-')
        id_saved_folder_base = '' if len(cp_id) == 1 else cp_id[0]
    elif args.bo_name:
        id_saved_folder_base = args.bo_name + '-'
    else:
        id_saved_folder_base = ''
    i_cp = int(args.from_checkpoint.split('-')[-1]) if args.from_checkpoint is not None else 0
    if not args.in_place and args.from_checkpoint is not None:
        i_cp += 1

    for script in range(1, args.nscripts):
        print(f'Script {script + 1}/{args.nscripts}')
        s = 'python bayesian-opt.py'
        # We only need to say the number of call and the check point
        # And the if in_place, we keep it
        s += f' --n-calls {args.n_calls}'
        s += f' --from-checkpoint {id_saved_folder_base}{i_cp}'
        if args.in_place:
            s += ' --in-place'
        else:
            i_cp += 1
        os.system(s)


if __name__ == '__main__':
    parser = Parser(argtype=ArgType.NScriptsBO)
    args = parser.parse_args()

    args = Args.preprocess.n_scripts_bo(args)
    main(args)
