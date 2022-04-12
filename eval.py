import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable

from .src.ReDWebNet import ReDWebNet_resnet50

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class YouTube3D():
    def __init__(self):
        self.weights = './models/YouTube3D/weights/data_model/src/exp/YTmixReD_dadlr1e-4_DIW_ReDWebNet_1e-6_bs4/models/model_iter_753000.bin'
        self.model = ReDWebNet_resnet50().to(device)
        self.model.load_state_dict(torch.load(self.weights))	

    def evaluate(self, img):
        img = np.asarray(Image.open(img)) / 255.0
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float().unsqueeze(0).to(device)
        img = Variable(img).to(device)
        
        with torch.no_grad():
            pred = self.model(img)
        pred = pred.squeeze().cpu().numpy()

        # min max normalization
        pred = (pred - pred.min()) / (pred.max() - pred.min())

        return pred
