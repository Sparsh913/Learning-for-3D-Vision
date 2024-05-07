import numpy as np
import argparse

import torch
from models import cls_model
from pointnet_pp import cls_model
from utils import create_dir, viz_cloud
from torch.nn.parallel import DistributedDataParallel as DDP

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_cls_class', type=int, default=3, help='The number of classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')
    parser.add_argument('--batch_size', type=int, default=32, help='The number of images in a batch.')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='model_epoch_0')
    parser.add_argument('--i', type=int, default=0, help="index of the object to visualize")

    parser.add_argument('--test_data', type=str, default='./data/cls/data_test.npy')
    parser.add_argument('--label_test', type=str, default='./data/cls/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--exp_name', type=str, default="exp", help='The name of the experiment')

    return parser


if __name__ == '__main__':
    torch.cuda.empty_cache()
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # ------ TO DO: Initialize Model for Classification Task ------
    # model = torch.nn.DataParallel(cls_model(num_classes=args.num_cls_class).to(args.device), device_ids=[0,1, 2, 3]).to(args.device)
    # model = DDP(cls_model(num_classes=args.num_cls_class), device_ids=[1])
    model = cls_model(num_classes=args.num_cls_class).to(args.device)
    # Load Model Checkpoint
    model_path = './checkpoints/cls/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        print(state_dict.keys())
        # append "module." to the keys of the state_dict at the start of the key
        # adding "module." is necessary because the model was saved without using DataParallel
        # state_dict = {k if 'module.' in k else 'module.'+k: v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
    model.eval()
    print ("successfully loaded checkpoint from {}".format(model_path))


    # Sample Points per Object
    ind = np.random.choice(10000,args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:]).to(args.device) # (953, 10000, 3)
    # Rotate the points in the test data by 45 degrees about the z-axis
    rotation_matrix_45 = torch.tensor([[np.cos(np.pi/4), -np.sin(np.pi/4), 0], [np.sin(np.pi/4), np.cos(np.pi/4), 0], [0, 0, 1]]).to(args.device).float()
    rotation_matrix_90 = torch.tensor([[np.cos(np.pi/2), -np.sin(np.pi/2), 0], [np.sin(np.pi/2), np.cos(np.pi/2), 0], [0, 0, 1]]).to(args.device).float()
    rotation_matrix_30 = torch.tensor([[np.cos(np.pi/6), -np.sin(np.pi/6), 0], [np.sin(np.pi/6), np.cos(np.pi/6), 0], [0, 0, 1]]).to(args.device).float()
    rotation_matrix_60 = torch.tensor([[np.cos(np.pi/3), -np.sin(np.pi/3), 0], [np.sin(np.pi/3), np.cos(np.pi/3), 0], [0, 0, 1]]).to(args.device).float()
    # test_data = torch.matmul(test_data, rotation_matrix_60) # (953, 10000, 3)
    
    label_test = torch.from_numpy(np.load(args.label_test)).to(args.device)
    print("test labels unique: ", torch.unique(label_test))
    torch.cuda.empty_cache()
    
    # ------ TO DO: Make Prediction ------
    
    data_loader = torch.split(test_data, args.batch_size) # (30, 32, 10000, 3)
    label_loader = torch.split(label_test, args.batch_size)
    labels_pred = []


    for data, label in zip(data_loader, label_loader):
        label_pred =  model(data)
        label_pred = torch.argmax(label_pred, -1)
        labels_pred.append(label_pred)

    # Compute Accuracy
    labels_pred = torch.cat(labels_pred)
    test_accuracy = labels_pred.eq(label_test.data).cpu().sum().item() / (label_test.shape[0])
    print (f"test accuracy: {test_accuracy}")

    # Visualize 3 random test point clouds and mention the predicted classes for each.
    n = len(test_data) # 953
    print("n: ", n)
    for i in (476, 700, 952, 406, 619, 726):
        print("shape of test_data[i]: ", test_data[i].shape)
        viz_cloud(test_data[i], f"output/q4/point_cloud_{i}.gif")
        print("Ground truth class for point cloud {}: {}".format(i, label_test[i]))
        print("Predicted class for point cloud {}: {}".format(i, labels_pred[i]))
    
    # labels are 0, 1, 2
    # visualize 3 point clouds where pred labels don't match test labels (one for each class)
    # j = 0
    # for i in range(n):
    #     if label_test[i] != labels_pred[i]:
    #         j += 1
    #         print("shape of test_data[i]: ", test_data[i].shape)
    #         viz_cloud(test_data[i], f"output/point_cloud_{i}.gif")
    #         print("Ground truth class for point cloud {}: {}".format(i, label_test[i]))
    #         print("Predicted class for point cloud {}: {}".format(i, labels_pred[i]))
    #         if j == 3:
    #             break
    
    # Find a point cloud where gt label is 2 and pred label is either 0 or 1
    # for i in range(n):
    #     if label_test[i] == 0 and labels_pred[i] != 0:
    #         print("shape of test_data[i]: ", test_data[i].shape)
    #         viz_cloud(test_data[i], f"output/point_cloud_{i}.gif")
    #         print("Ground truth class for point cloud {}: {}".format(i, label_test[i]))
    #         print("Predicted class for point cloud {}: {}".format(i, labels_pred[i]))
    #         break