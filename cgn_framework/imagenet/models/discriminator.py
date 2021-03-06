import numpy as np
import torch
from torch import embedding, nn


from utils import init_net


class DiscLin(nn.Module):
    def __init__(self, n_classes=1000, ndf=12, img_shape=[3, 256, 256]):
        super(DiscLin, self).__init__()

        self.label_embedding = nn.Embedding(n_classes, n_classes)

        self.model = nn.Sequential(
            nn.Linear(n_classes + int(np.prod(img_shape)), ndf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(ndf, ndf),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(ndf, ndf),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(ndf, 1),
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        self.label_embedding(labels)  # ToDo: figure out concatenation
        embedding = embedding.reshape(-1, 1, 1, 1)
        embedding = embedding.repeat(1, 1, *img.shape[-2:])
        d_in = torch.cat((img.view(img.size(0), -1), embedding), -1)
        validity = self.model(d_in)
        return validity.squeeze()
