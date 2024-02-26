import argparse
import time
import torch
from model import SingleViewto3D
from r2n2_custom import R2N2
from  pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
import dataset_location
import pytorch3d
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.ops import knn_points
import mcubes
import utils_vox
import matplotlib.pyplot as plt
from render import mesh_to_image, points_to_image

# GradCAM imports -> remove later
# from torchvision.io.image import read_image
# from torchvision.transforms.functional import normalize, resize, to_pil_image
# from torchvision.models import resnet18
# from torchcam.methods import SmoothGradCAMpp

def get_args_parser():
    parser = argparse.ArgumentParser('Singleto3D', add_help=False)
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--vis_freq', default=1000, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh'], type=str)
    parser.add_argument('--n_points', default=1000, type=int)
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=0.1, type=float)  
    parser.add_argument('--load_checkpoint', action='store_true')  
    parser.add_argument('--device', default='cuda', type=str) 
    parser.add_argument('--load_feat', action='store_true') 
    return parser

def preprocess(feed_dict, args):
    for k in ['images']:
        feed_dict[k] = feed_dict[k].to(args.device)

    images = feed_dict['images'].squeeze(1)
    mesh = feed_dict['mesh']
    # mesh = feed_dict['voxel_coords']
    if args.load_feat:
        images = torch.stack(feed_dict['feats']).to(args.device)

    return images, mesh

def save_plot(thresholds, avg_f1_score, args):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(thresholds, avg_f1_score, marker='o')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1-score')
    ax.set_title(f'Evaluation {args.type}')
    plt.savefig(f'eval_{args.type}', bbox_inches='tight')


def compute_sampling_metrics(pred_points, gt_points, thresholds, eps=1e-8):
    metrics = {}
    lengths_pred = torch.full(
        (pred_points.shape[0],), pred_points.shape[1], dtype=torch.int64, device=pred_points.device
    )
    lengths_gt = torch.full(
        (gt_points.shape[0],), gt_points.shape[1], dtype=torch.int64, device=gt_points.device
    )

    # For each predicted point, find its neareast-neighbor GT point
    knn_pred = knn_points(pred_points, gt_points, lengths1=lengths_pred, lengths2=lengths_gt, K=1)
    # Compute L1 and L2 distances between each pred point and its nearest GT
    pred_to_gt_dists2 = knn_pred.dists[..., 0]  # (N, S)
    pred_to_gt_dists = pred_to_gt_dists2.sqrt()  # (N, S)

    # For each GT point, find its nearest-neighbor predicted point
    knn_gt = knn_points(gt_points, pred_points, lengths1=lengths_gt, lengths2=lengths_pred, K=1)
    # Compute L1 and L2 dists between each GT point and its nearest pred point
    gt_to_pred_dists2 = knn_gt.dists[..., 0]  # (N, S)
    gt_to_pred_dists = gt_to_pred_dists2.sqrt()  # (N, S)

    # Compute precision, recall, and F1 based on L2 distances
    for t in thresholds:
        precision = 100.0 * (pred_to_gt_dists < t).float().mean(dim=1)
        recall = 100.0 * (gt_to_pred_dists < t).float().mean(dim=1)
        f1 = (2.0 * precision * recall) / (precision + recall + eps)
        metrics["Precision@%f" % t] = precision
        metrics["Recall@%f" % t] = recall
        metrics["F1@%f" % t] = f1

    # Move all metrics to CPU
    metrics = {k: v.cpu() for k, v in metrics.items()}
    return metrics

def evaluate(predictions, mesh_gt, thresholds, args):
    if args.type == "vox":
        voxels_src = predictions
        H,W,D = voxels_src.shape[2:]
        vertices_src, faces_src = mcubes.marching_cubes(voxels_src.detach().cpu().squeeze().numpy(), isovalue=0.3)
        vertices_src = torch.tensor(vertices_src).float()
        faces_src = torch.tensor(faces_src.astype(int))
        mesh_src = pytorch3d.structures.Meshes([vertices_src], [faces_src])
        pred_points = sample_points_from_meshes(mesh_src, args.n_points)
        pred_points = utils_vox.Mem2Ref(pred_points, H, W, D)
    elif args.type == "point":
        pred_points = predictions.cpu()
    elif args.type == "mesh":
        pred_points = sample_points_from_meshes(predictions, args.n_points).cpu()

    gt_points = sample_points_from_meshes(mesh_gt, args.n_points)
    
    metrics = compute_sampling_metrics(pred_points, gt_points, thresholds)
    return metrics, gt_points



def evaluate_model(args):
    r2n2_dataset = R2N2("test", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True, return_feats=args.load_feat)

    loader = torch.utils.data.DataLoader(
        r2n2_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_batched_R2N2,
        pin_memory=True,
        drop_last=True)
    eval_loader = iter(loader)

    model = SingleViewto3D(args)
    model.to(args.device)
    model.eval()
    # print("Model encoder: ", model.encoder)
    # print("Model decoder: ", model.decoder)
    # temp_model = model.encoder.eval()
    # for i in range(500):
    #     img = read_image(f'outputs/q2/{i}_rgb_gt.png')
    #     # remove alpha channel
    #     img = img[:3]
    #     print("img: ", img.shape)
    #     input_tensor = normalize(resize(img, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).to(args.device)
    #     with SmoothGradCAMpp(temp_model) as cam_extractor:
    #         # Preprocess your data and feed it to the temp_model
    #         out = temp_model(input_tensor.unsqueeze(0))
    #         # Retrieve the CAM by passing the class index and the model output
    #         activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
            
    #     import matplotlib.pyplot as plt
    #     from torchcam.utils import overlay_mask
    #     # Visualize the raw CAM
    #     # plt.imshow(activation_map[0].squeeze(0).numpy()); plt.axis('off'); plt.tight_layout(); plt.show()
    #     # Resize the CAM and overlay it
    #     result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
    #     # Display it
    #     # plt.imshow(result); plt.axis('off'); plt.tight_layout(); plt.show()
    #     # plt.show()
    #     # Convert the image to rgb and save it
    #     result = result.convert("RGB")
    #     result.save(f'outputs/q2/{i}_cam.png')
    #     # plt.imsave('outputs/q2/gradcam/0_cam.png', result)
    
    start_iter = 0
    start_time = time.time()

    thresholds = [0.01, 0.02, 0.03, 0.04, 0.05]

    avg_f1_score_05 = []
    avg_f1_score = []
    avg_p_score = []
    avg_r_score = []

    if args.load_checkpoint:
        print('args.type: ', args.type)
        print('args.n_points: ', args.n_points)
        checkpoint = torch.load(f'checkpoint_{args.type}.pth')
        # try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Succesfully loaded iter {start_iter}")
        # except RuntimeError as e:
        # print('Ignoring "' + str(e) + '"')
    
    print("Starting evaluating !")
    max_iter = len(eval_loader)
    try:
        for step in range(start_iter, max_iter):
            iter_start_time = time.time()

            read_start_time = time.time()

            feed_dict = next(eval_loader) # feed_dict -> dict_keys(['synset_id', 'model_id', 'verts', 'faces', 'label', 'images', 'R', 'T', 'K', 'voxel_coords', 'voxels', 'mesh'])
            # print("feed_dict: ", feed_dict.keys())
            voxel_coords = feed_dict['voxel_coords']
            print("voxel_coords: ", voxel_coords[0].shape)
            images_gt, mesh_gt = preprocess(feed_dict, args)
            print("mesh_gt: ", mesh_gt)
            # print("mesh_get")
            print("images_gt shape: ", images_gt.shape)

            read_time = time.time() - read_start_time

            predictions = model(images_gt, args) # b x 1 x 32 x 32 x 32
            # print("predictions shape: ", predictions.shape)

            if args.type == "vox":
                predictions = predictions.permute(0,1,4,3,2) # b x 1 x 32 x 32 x 32 

            metrics, gt_points = evaluate(predictions, mesh_gt, thresholds, args)

            # TODO:
            # if (step % args.vis_freq) == 0:
            #     # visualization block
            # mesh_to_image(mesh_gt, out_path=f'outputs/q2/{step}_mesh_gt', dist=2, elev = 0) # Note that mesh_gt is a Pytorch3d mesh
            if args.type == "vox":
                H,W, D = predictions.shape[2:]
                mesh_pred = mcubes.marching_cubes(predictions.detach().cpu().squeeze().numpy(), isovalue=0.3)
                mesh_to_image(mesh_pred, out_path=f'outputs/q2/voxel/{step}_mesh_pred', dist=68, elev = 24)
            elif args.type == "point":
                points_to_image(predictions, out_path=f'outputs/q3/3.2/{step}_point_pred', num_views=12)
                # points_to_image(gt_points, out_path=f'outputs/q3/{step}_point_gt', num_views=12)
            elif args.type == "mesh":
                mesh_to_image(predictions, out_path=f'outputs/q2/mesh/w_smooth_0.8/{step}_mesh_pred', dist=3, elev = 18)
            plt.imsave(f'outputs/q2/{step}_rgb_gt.png', images_gt.squeeze().cpu().numpy())
            #     #  rend = 
            #     plt.imsave(f'vis/{step}_{args.type}.png', rend)
        

            total_time = time.time() - start_time
            iter_time = time.time() - iter_start_time

            f1_05 = metrics['F1@0.050000']
            avg_f1_score_05.append(f1_05)
            avg_p_score.append(torch.tensor([metrics["Precision@%f" % t] for t in thresholds]))
            avg_r_score.append(torch.tensor([metrics["Recall@%f" % t] for t in thresholds]))
            avg_f1_score.append(torch.tensor([metrics["F1@%f" % t] for t in thresholds]))

            print("[%4d/%4d]; ttime: %.0f (%.2f, %.2f); F1@0.05: %.3f; Avg F1@0.05: %.3f" % (step, max_iter, total_time, read_time, iter_time, f1_05, torch.tensor(avg_f1_score_05).mean()))
            
    except:
        print("Ignoring error in Empty meshes")

    # avg_f1_score = torch.stack(avg_f1_score).mean(0)

    # save_plot(thresholds, avg_f1_score,  args)
    print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Singleto3D', parents=[get_args_parser()])
    args = parser.parse_args()
    evaluate_model(args)
