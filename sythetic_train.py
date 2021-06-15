import os
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import numpy as np
from datasets.synthetic_burst_train_set import SyntheticBurst
from models.dbsr import DeepBurstSR
from utils.metrics import L2
from datetime import datetime
import cv2

      

def save_checkpoint(epoch, model, loss, save_dir):
    state = {
        'model': model.state_dict(),
        'loss': loss,
        'epoch': epoch,
    }
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    torch.save(state, f'{save_dir}/ckpt.pth')


def log_results(epoch, burst, frame_gt, outputs, save_dir='./logs'):
    burst = burst.cpu().numpy()
    frame_gt = frame_gt.cpu().numpy()
    outputs = outputs.cpu().numpy()
    now = datetime.now().time().strftime("%H-%M-%S") # time object
    fileprefix = save_dir + f'/Epoch-{epoch}'
    for i in range(burst.shape[0]):
        filename = fileprefix + f'-item-{i}'
        cv2.imwrite(filename+f'-output.jpg', outputs[i].transpose(1, 2, 0)*255)
        cv2.imwrite(filename+f'-target.jpg', frame_gt[i].transpose(1, 2, 0)*255)


def train(epoch, model, data_loader, optimizer, criterion, device='cuda'):
    model.train()
    train_loss = 0
    with tqdm(data_loader, unit="batch") as tepoch:
        for _, (burst, frame_gt, flow_vectors, _) in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch}")
            
            burst, frame_gt, flow_vectors = burst.to(device), frame_gt.to(device), flow_vectors.to(device)

            optimizer.zero_grad()

            outputs = model(burst, flow_vectors)
            loss = criterion(outputs, frame_gt)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            tepoch.set_postfix(loss=loss.item())

    return model


def test(epoch, model, data_loader, optimizer, criterion, best_l2=0, device='cuda', save_dir='./checkpoint'):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        with tqdm(data_loader, unit="batch") as tepoch:
            for batch_idx, (burst, frame_gt, flow_vectors, _) in enumerate(tepoch):
                tepoch.set_description(f"Testing: ")
                
                burst, frame_gt, flow_vectors = burst.to(device), frame_gt.to(device), flow_vectors.to(device)
                outputs = model(burst, flow_vectors)
                loss = criterion(outputs, frame_gt)

                test_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())

                if batch_idx%100==0:
                    log_results(epoch, burst, frame_gt, outputs, save_dir='./logs')

    # Save checkpoint.
    if test_loss < best_l2:
        save_checkpoint(epoch, model, test_loss, save_dir)
        best_l2 = test_loss

    return best_l2



# Parameters
best_l2 = 10000000  # best test l2
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
device = 'cuda'
batch_size = 5


# Data
print('==> Preparing data..')
imagenet_train = ImageNet('./imagenet_root/', 'train')
imagenet_test = ImageNet('./imagenet_root/', 'val')
train_set = SyntheticBurst(imagenet_train)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
test_set = SyntheticBurst(imagenet_test)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)


# Model
model = DeepBurstSR()
criterion = L2(boundary_ignore=20)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

model = model.to(device)
criterion = criterion.to(device)

for epoch in range(start_epoch, 500):
    model = train(epoch, model, train_loader, optimizer, criterion)
    best_l2 = test(epoch, model, test_loader, optimizer, criterion, best_l2, save_dir='./dbsr')
    scheduler.step()
