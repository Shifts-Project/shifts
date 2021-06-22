import argparse
from partitioner import Partitioner, Config

commandLineParser = argparse.ArgumentParser(description='Partition data.')
commandLineParser.add_argument('data_path', type=str, help='Path to data')
commandLineParser.add_argument('climate_info_path', type=str, help='Path to climate information')
commandLineParser.add_argument('save_path', type=str, help='Path to save partitioned data files')
commandLineParser.add_argument('--time_splits', nargs=4, type=float, default=[0.6, 0.1, 0.15, 0.15], help='Time splits')
commandLineParser.add_argument('--climate_splits', nargs=3, type=int, default=[3, 1, 1], help='Climate splits')
commandLineParser.add_argument('--in_domain_splits', nargs=3, type=float, default=[0.7, 0.15, 0.15], help='Fraction of training, dev and eval data to use for in-domain')
commandLineParser.add_argument('--no_meta', type=str, default='yes', choices=['yes', 'no'], help='yes or no for removing meta data')
commandLineParser.add_argument('--eval_dev_overlap', type=str, default='yes', choices=['yes', 'no'], help='yes or no for climate overlap between dev and eval')


def main():
    '''Partitions tabular weather data for distributional shift'''
    args = commandLineParser.parse_args()

    eval_dev_overlap = False
    if args.eval_dev_overlap == 'yes':
        eval_dev_overlap = True

    # Load the configurable parameters
    config = Config(args.time_splits, args.climate_splits, args.in_domain_splits, eval_dev_overlap=eval_dev_overlap)

    # Partition the raw weather data
    partitioner = Partitioner(args.data_path, args.climate_info_path, config)
    print()
    # Print number of data points in each data split
    for name, df in partitioner.dfs_to_save.items():
        print(name, df.shape[0])

    print()
    # Save all files
    partitioner.save(args.save_path, args.no_meta)

if __name__ == '__main__':
    main()