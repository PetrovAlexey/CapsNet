import argparse

def get_setup_args_train():
    parser = argparse.ArgumentParser('train')

    parser.add_argument('--dataset',
                        type=str,
                        default=r"MNIST")

    parser.add_argument('--data_dir',
                        type=str,
                        default=r"./data/")

    parser.add_argument('--model_name',
                        type=str,
                        default="resnet18")

    parser.add_argument('--load',
                        type=bool,
                        default="false")

    args = parser.parse_args()

    return args

def get_setup_args_test():
    parser = argparse.ArgumentParser('test')

    parser.add_argument('--dataset',
                        type=str,
                        default=r"MNIST")

    parser.add_argument('--model_name',
                        type=str,
                        default=r"resnet18")

    args = parser.parse_args()

    return args