import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms

from sklearn.neighbors import NearestNeighbors

resnet18 = models.resnet18()

data_transformer = {
    'eval': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
}

print("Initialize dataset")

image_datasets = {
    'eval': datasets.ImageFolder("~/Desktop/Kaggle/h-and-m-personalized-fashion-recommendations/images", data_transformer['eval'])
}
dataloader_dicts = {
    'eval': torch.utils.data.DataLoader(image_datasets['eval'], batch_size=100, shuffle=False, num_workers=1)
}

res = []
cnt = 0

for inputs, label in dataloader_dicts['eval']:
    print(f"Loading batch {cnt}")
    outputs = resnet18(inputs)
    for i in range(0, 100):
        embedding = outputs[i]
        res.append(embedding.cpu().detach().numpy())
    cnt += 1
    if cnt % 100 == 0:
        num = cnt // 100
        print(f"saving the batch of embeddings {num}")
        filename = f"embeddings_{num}.pt"
        torch.save(res, filename)
        res = []

nbrs = NearestNeighbors(n_neighbors=12, algorithm='ball_tree').fit(all_embeddings)

import time

result = []

for i in range(0, 105):
    print(f"Start processing batch {i}")
    start = time.time()
    sub_group = all_embeddings[i*1000:(i+1)*1000]
    _, neighbors = nbrs.kneighbors(sub_group)
    result.append(neighbors)
    end = time.time()
    print(f"Finish processing batch {i}, time consumed {end - start}")
