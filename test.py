import os.path as osp
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch

model_path = 'models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
device = torch.device('cpu')  # Use CPU

test_img_folder = 'LR/*'

# Load the model
model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path, map_location=device), strict=True)  # Ensure model is loaded to CPU
model.eval()
model = model.to(device)

print(f'Model path {model_path}. \nTesting...')

idx = 0
for path in glob.glob(test_img_folder):
    idx += 1
    base = osp.splitext(osp.basename(path))[0]
    print(idx, base)
    
    # Read and preprocess the image
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0).to(device)  # Ensure tensor is on the CPU

    # Perform inference
    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().clamp_(0, 1).cpu().numpy()
    
    # Postprocess and save the output
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    cv2.imwrite(f'results/{base}_rlt.png', output)

print("Testing completed.")
