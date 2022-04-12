import cv2
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable

from .src.ReDWebNet import ReDWebNet_resnet50, resNet_data_preprocess

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class YouTube3D():
    def __init__(self):
        self.weights = './models/YouTube3D/weights/data_model/src/exp/YTmixReD_dadlr1e-4_DIW_ReDWebNet_1e-6_bs4/models/model_iter_753000.bin'
        self.model = ReDWebNet_resnet50().to(device)
        self.model.load_state_dict(torch.load(self.weights))	

    def evaluate(self, img):
        new_height, new_width = 384, 384

        color = cv2.imread(img)
        original_height, original_width, _ = color.shape
        color = cv2.resize(color, (new_width, new_height))
        color = color.transpose(2, 0, 1).astype(np.float32) / 255.
        color = resNet_data_preprocess(color)
        img = torch.from_numpy(color).float().unsqueeze(0).to(device)
        img = Variable(img).to(device)
        
        with torch.no_grad():
            pred = self.model(img)

        pred = pred.squeeze().cpu().numpy()
        import ipdb; ipdb.set_trace()
        # min max normalization
        pred = (pred - pred.min()) / (pred.max() - pred.min())
        
        # convert depth to disparity
        pred = 1.0 / (pred + 0.1)

        # min max normalization
        pred = (pred - pred.min()) / (pred.max() - pred.min())

        # resize to original size
        pred = cv2.resize(pred, (original_width, original_height))

        return pred
