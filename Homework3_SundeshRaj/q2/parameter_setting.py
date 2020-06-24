import argparse
import torch


##### Modify the parameters in this file to change different settings

def parameter_setting():
    # Training settings
    parser = argparse.ArgumentParser()

    ## Hyper-parameters used in the experiment (You may need to add or modify the parameters)
    parser.add_argument("--batch_size", default=32, help="batch size")
    parser.add_argument("--lr", default=0.01, help="learning rate")
    parser.add_argument("--root", default="./", help="root")
    parser.add_argument("--learningRate", default=0.01)
    parser.add_argument("--momentum", default=0.9)
    parser.add_argument("--wd", default=0.09)
    parser.add_argument("--print_interval", default=100)

    parser.add_argument("--epoch_number", default=100, help="epoch number")
    parser.add_argument("--dataset_name", default="fashion_mnist", help="cifar10; fashion_mnist")
    parser.add_argument("--model", default="onelayer", help="mlp, cnn, onelayer")
    parser.add_argument("--input_channel", default=1)
    parser.add_argument("--defaultNumClasses", default=1)
    parser.add_argument("--defNumClasses", default=10)

    ## cifar10 (You may need to add or modify the parameters)
    parser.add_argument("--image_cifar10_width", default=32, help="width of the cifar10 image")
    parser.add_argument("--image_cifar10_height", default=32, help="height of the cifar10 image")
    parser.add_argument("--class_number_cifar10", default=10, help="number of classes in CIFAR10")
    parser.add_argument("--dropout", default=0.0)

    ## fashion mnist (You may need to add or modify the parameters)
    parser.add_argument("--image_fashion_mnist_width", default=28, help="width of the fashion mnist image")
    parser.add_argument("--image_fashion_mnist_height", default=28, help="height of the fashion mnist image")
    parser.add_argument("--class_number_fashion_mnist", default=10, help="number of classes in fashion mnist")
    parser.add_argument("--visual_flag", default=False,
                        help="To determine whether to visualize the weights of neural network or not")

    parser.add_argument("--log_interval", default=1, help="interval to print results to screen")
    parser.add_argument("--save_weight_interval", default=5,
                        help="interval to save network visualization figure to disk")

    ## input folder
    parser.add_argument("--data_path", default="./data")

    ## output folder
    parser.add_argument("--output_folder", default="./output")

    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    return args
