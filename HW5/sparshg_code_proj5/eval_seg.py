import numpy as np
import argparse

import torch
from models import seg_model
from data_loader import get_data_loader
from utils import create_dir, viz_seg


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_seg_class', type=int, default=6, help='The number of segmentation classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')
    parser.add_argument('--batch_size', type=int, default=32, help='The number of images in a batch.')
    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='model_epoch_0')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/seg/data_test.npy')
    parser.add_argument('--label_test', type=str, default='./data/seg/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Segmentation Task  ------
    model = seg_model(num_seg_classes=args.num_seg_class).to(args.device)
    
    # Load Model Checkpoint
    model_path = './checkpoints/seg/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))


    # Sample Points per Object
    ind = np.random.choice(10000,args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:]).to(args.device)
    # rotation_matrix_45 = torch.tensor([[0.7071, -0.7071, 0], [0.7071, 0.7071, 0], [0, 0, 1]]).to(args.device)
    # rotation_matrix_90 = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]]).to(args.device).float()
    # rotation_matrix_30 = torch.tensor([[np.cos(np.pi/6), -np.sin(np.pi/6), 0], [np.sin(np.pi/6), np.cos(np.pi/6), 0], [0, 0, 1]]).to(args.device).float()
    # rotation_matrix_60 = torch.tensor([[np.cos(np.pi/3), -np.sin(np.pi/3), 0], [np.sin(np.pi/3), np.cos(np.pi/3), 0], [0, 0, 1]]).to(args.device).float()
    # test_data = torch.matmul(test_data, rotation_matrix_90)
    
    label_test = torch.from_numpy((np.load(args.label_test))[:,ind]).to(args.device)
    print("label_test shape: ", label_test.shape)
    print("unique test labels: ", torch.unique(label_test))

    # ------ TO DO: Make Prediction ------
    data_loader = torch.split(test_data, args.batch_size)
    label_loader = torch.split(label_test, args.batch_size)
    labels_pred = []
    for data, label in zip(data_loader, label_loader):
        label_pred =  model(data).to(args.device)
        label_pred = torch.argmax(label_pred, 1)
        labels_pred.append(label_pred)
    
    labels_pred = torch.cat(labels_pred).to(args.device)
    print("labels_pred shape: ", labels_pred.shape)
    print("unique pred labels: ", torch.unique(labels_pred))
    # label_pred = model(test_data.to(args.device))

    test_accuracy = labels_pred.eq(label_test.data).cpu().sum().item() / (label_test.reshape((-1,1)).shape[0])
    print ("test accuracy: {}".format(test_accuracy))

    # Visualize Segmentation Result (Pred VS Ground Truth)
    viz_seg(test_data[args.i], label_test[args.i], "{}/gt_{}.gif".format(args.output_dir, args.exp_name), args.device)
    viz_seg(test_data[args.i], label_pred[args.i], "{}/pred_{}.gif".format(args.output_dir, args.exp_name), args.device)

    n = len(test_data)
    print("n: ", n)
    # for i in range(n):
    #     if label_test[i] != label_pred[i]:
    #         viz_seg(test_data[i], label_test[i], "{}/gt_{}.gif".format(args.output_dir, i), args.device)
    #         viz_seg(test_data[i], label_pred[i], "{}/pred_{}.gif".format(args.output_dir, i), args.device)
    #         break
    
    # visualize point cloud if any of the label value in the tensors pred labels and test labels don't match.
    # j = 0
    # for i in range(n):
    #     if not torch.equal(label_test[i], labels_pred[i]):
    #         j += 1
    #         viz_seg(test_data[i], label_test[i], "{}/gt_{}.gif".format(args.output_dir, i), args.device)
    #         viz_seg(test_data[i], labels_pred[i], "{}/pred_{}.gif".format(args.output_dir, i), args.device)
    #         # break
    #         if j == 3:
    #             break
    
    # pred labels have 1000 values, if more than 50 % are different than test labels, visualize the point cloud
    j = 0
    for i in (10, 300, 600):
        # if torch.sum(label_test[i] == labels_pred[i]) > 0.99*args.num_points:
        #     j += 1
        print("i: ", i)
        viz_seg(test_data[i], label_test[i], "{}/gt_{}.gif".format(args.output_dir, i), args.device)
        viz_seg(test_data[i], labels_pred[i], "{}/pred_{}.gif".format(args.output_dir, i), args.device)
        print("accuracy for point cloud {}: {}".format(i, torch.sum(label_test[i] == labels_pred[i])/args.num_points))
            # break
            # if j == 3:
            #     break