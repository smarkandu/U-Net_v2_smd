import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import cv2

from lib.pvt import PolypPVT
from utils.dataloader import test_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    parser.add_argument('--pth_path', type=str,
                        default='PolypSeg/pvt_pth/PolypPVT.pth',
                        help='path to PolypPVT checkpoint (.pth)')
    parser.add_argument('--data_root', type=str,
                        default='TestDataset',
                        help='root containing CVC-300, CVC-ClinicDB, Kvasir, CVC-ColonDB, ETIS-LaribPolypDB')
    parser.add_argument('--save_root', type=str,
                        default='result_map/PolypPVT',
                        help='folder to save predictions')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # --- build model & load weights ---
    model = PolypPVT().to(device)
    ckpt = torch.load(args.pth_path, map_location=device)
    # handle DataParallel checkpoints
    if any(k.startswith('module.') for k in ckpt.keys()):
        ckpt = {k.replace('module.', '', 1): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=True)
    model.eval()

    datasets = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']

    with torch.no_grad():
        for ds in datasets:
            data_path = os.path.join(args.data_root, ds)
            image_root = os.path.join(data_path, 'images')
            gt_root    = os.path.join(data_path, 'masks')
            save_path  = os.path.join(args.save_root, ds)
            os.makedirs(save_path, exist_ok=True)

            if not os.path.isdir(image_root) or not os.path.isdir(gt_root):
                raise FileNotFoundError(f"Missing {image_root} or {gt_root}")

            num_imgs = len(os.listdir(gt_root))
            loader = test_dataset(image_root, gt_root, args.testsize)

            for _ in range(num_imgs):
                image, gt, name = loader.load_data()   # image: torch tensor (B=1,C,H,W); gt: np array (H,W)
                gt = np.asarray(gt, np.float32)
                gt /= (gt.max() + 1e-8)

                image = image.to(device)
                # PolypPVT returns two outputs (P1, P2)
                P1, P2 = model(image)
                res = P1 + P2
                res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
                res = torch.sigmoid(res).cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)  # normalize to [0,1]
                out = (res * 255).astype(np.uint8)

                cv2.imwrite(os.path.join(save_path, name), out)

            print(ds, 'Finish!')

if __name__ == '__main__':
    main()
