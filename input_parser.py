"""This module is responsible for receiving and parsing parameters from the command line"""
import argparse


def parse_arguments():
    """
    Parses and validates command line input. Throws an error on invalid inputs

    :returns an object of packed argparse arguments
     """
    parser = argparse.ArgumentParser()
    # Parse k
    parser.add_argument("k", type=int)
    # Parse n
    parser.add_argument("n", type=int)
    random_parser = parser.add_mutually_exclusive_group(required=False)
    # Parse random flag
    random_parser.add_argument("--Random", dest="random", action="store_true")
    random_parser.add_argument("--no-Random", dest="random", action="store_false")
    # Value of the random flag by default
    parser.set_defaults(random=True)
    args = parser.parse_args()
    # If random-flag is set to True, the rest of the parameters are not used
    if args.random:
        return args
    # Sanity check
    if args.k < 1 or args.n < 1 or args.k > args.n:
        parser.error("Invalid value of k or/and n")

    return args
