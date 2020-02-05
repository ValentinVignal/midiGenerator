"""
To create summary and images from a search_result of bayesian optimization
"""
from termcolor import cprint

from src import BayesianOpt as BO
from src import Args
from src.Args import Parser, ArgType


def main(args):
    search_result, dimensions = BO.load.from_checkpoint(args.folder / 'checkpoint')
    BO.save.save_search_result(
        search_result=search_result,
        dimensions=dimensions,
        folder_path=args.folder
    )
    cprint('---------- Done ----------', 'grey', 'on_green')


if __name__ == '__main__':
    parser = Parser(argtype=ArgType.HPSummary)
    args = parser.parse_args()

    args = Args.preprocess.hp_summary(args)
    main(args)

