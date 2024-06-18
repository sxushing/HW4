import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
img_path = './input/test.png'
input_image = Image.open(img_path).convert('RGB')
transform = transforms.Compose([transforms.ToTensor()])
tensor = transform(input_image).unsqueeze(0)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        return x


model = Model()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 200
for epoch in range(epochs):
    epoch_loss = 0
    rand = torch.rand(1)
    optimizer.zero_grad()
    output = model(
        (tensor+torch.randn_like(tensor)*rand)/(1+rand))
    loss = criterion(output, tensor)
    loss.backward()
    optimizer.step()
    epoch_loss += loss.item()
    if epoch % 20 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss: {epoch_loss:.4f}')
        output_image = transforms.ToPILImage()(output.squeeze())
        output_image.save(f'./output/epoch_{epoch}.jpg')
        features_i = tensor.squeeze().permute(1, 2, 0).numpy()
        features_o = output.detach().squeeze().permute(1, 2, 0).numpy()
        mu_i = np.mean(features_i, axis=0)
        mu_o = np.mean(features_o, axis=0)
        sigma_i = np.cov(features_i[0], rowvar=True)
        sigma_o = np.cov(features_o[0], rowvar=True)
        square_distance = np.sum((mu_i - mu_o) ** 2)
        covariance_mean = linalg.sqrtm(sigma_i.dot(sigma_o))
        if np.iscomplexobj(covariance_mean):
            covariance_mean = covariance_mean.real
        fid = square_distance + np.trace(sigma_i + sigma_o - 2*covariance_mean)
        print(f'FID: {fid:.4f}')


transforms.ToPILImage()(output.squeeze()).save(f'./output/result.jpg')
