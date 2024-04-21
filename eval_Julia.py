import json
import os.path
import torch
from collections import OrderedDict
from Data_Generation.Julia_Set.fractal import Point, exec_command
from DataLoader.Data_Julia import get_data_Julia_loader
import torch.nn as nn
from PIL import Image
import cv2
import lpips
import numpy as np
from skimage import metrics
# String formatting functions
Point.__str__ = lambda self: "x".join(map(str, self))
basic_str = lambda el: str(el).lstrip("(").rstrip(")")  # Complex or model name
item_str = lambda k, v: (basic_str(v) if k in ["model", "c"] else
                         "{}={}".format(k, v))
filename = lambda kwargs: "_".join(item_str(*pair) for pair in kwargs.items())


def test_lpips(true_file_path, file_path1):
    # 假设您已经有了要计算LPIPS距离的两张图片 image1 和 image2
    # 加载图像文件
    image1 = Image.open(true_file_path).convert("RGB").resize((512, 512))
    image2 = Image.open(file_path1).convert("RGB").resize((512, 512))

    # 加载预训练的LPIPS模型
    lpips_model = lpips.LPIPS(net="alex")

    # 将图像转换为PyTorch的Tensor格式
    image1_tensor = torch.tensor(np.array(image1)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    image2_tensor = torch.tensor(np.array(image2)).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    # 使用LPIPS模型计算距离
    distance = lpips_model(image1_tensor, image2_tensor)

    print("LPIPS distance:", distance.item())
    return distance.item()

def test_SSIM(image_path1, image_path2):
    # Load the images
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)

    # Convert images to grayscale
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Compute SSIM
    ssim_score = metrics.structural_similarity(gray_image1, gray_image2)

    print("SSIM score:", ssim_score)
    return ssim_score
def rmse(predictions, targets):
    return torch.sqrt((predictions - targets)**2)

def test():
    device = 'cuda'
    _, test_loader = get_data_Julia_loader()
    model_list = []

    # ###############
    checkpoint = "checkpoints/Dense/BEST_checkpoint_fractal_random_1_cap_per_img_1_min_word_freq.pth.tar"
    checkpoint = torch.load(checkpoint)
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()
    model_list.append(encoder)
    # ####################
    with open("Data_Split/Julia_Set/test_data_random.json", 'r', encoding='utf-8') as f:
        true_data = json.load(f)

    test_criterion = nn.L1Loss(reduction='none')

    for model in model_list:
        model.eval()
        total_loss = 0.0
        RMSE_loss = 0.0

        save_dir = 'Test_Generate_Image/Dense/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        RMSE_list = []
        lpips_list = []
        SSMI_list = []
        real_list = []
        image_list = []
        zoom_list = []
        depth_list = []

        with torch.no_grad():
            for (batch, meta) in enumerate(test_loader):
                input = meta['image']
                label = meta['label']
                input = input.to(device)
                label = label.to(device)
                _, pre,_ = model(input)
                real = pre[0, 0].item()
                image = pre[0, 1].item()
                depth = int(pre[0, 2].item())
                zoom = pre[0, 3].item()
                real_list.append(real)
                image_list.append(image)
                depth_list.append(depth)
                zoom_list.append(zoom)
                temp = OrderedDict([("model", "julia"),
                                    ("c", complex(real, image)),
                                    ("size", Point(512, 512)),
                                    ("depth", depth),
                                    ("zoom", zoom), ])
                temp['output'] = save_dir + '/' + str(batch) + '_{}.png'.format(
                    filename(temp))

                exec_command(temp)
                real_path ="Data_Generation/Julia_Set/Data/" + true_data[batch]
                save_path = temp['output']
                print(real_path, save_path)
                lpip = test_lpips(real_path, save_path)
                lpips_list.append(lpip)
                ssmi = test_SSIM(real_path, save_path)
                SSMI_list.append(ssmi)
                loss = test_criterion(pre, label)
                total_loss += loss
                one_loss = rmse(pre, label)
                RMSE_loss = RMSE_loss + one_loss
                RMSE_list.append(one_loss)
            print('LPIPS：', sum(lpips_list) / len(lpips_list))

            print('SSMI：', sum(SSMI_list) / len(SSMI_list))
        print("MAE:",total_loss)
        print("MAE:",total_loss/500)
        print("RMSE:", RMSE_loss)
        print("RMSE:", RMSE_loss / 500)
if __name__ == '__main__':
    test()
