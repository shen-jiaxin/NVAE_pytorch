import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import platform
import os
import math
from nvae import NvaeModel


EPOCH = 50
BATCH_SIZE = 128
LR = 0.0003
NUM_WORKERS = 0 if platform.system() == 'Windows' else 4
datasets = ['MNIST', 'CIFAR10', 'CelebA']
dataset = datasets[1]
NUM_IN_CHANNEL = 3  # 1 for gray images and 3 for RGB images
NUM_IN_SIZE = (32, 32)  # target size of resizing, (H, W)
LATENT_Z_SIZE = (64, 4, 4)  # target dimensions of latent variable with highest level, (C, H, W)
DOUBLE_GROUP = True  # option for doubling groups of latent variables

save_pics_path = './pics/' + dataset
dataset_path = r'D:\Data\SoftwareSave\Python\Datasets'
if not os.path.exists(save_pics_path):
    os.makedirs(save_pics_path)

transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize((NUM_IN_SIZE[0], NUM_IN_SIZE[1])),
     torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

if dataset == 'MNIST':
    NUM_IN_CHANNEL = 1
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((NUM_IN_SIZE[0], NUM_IN_SIZE[1])),
         torchvision.transforms.ToTensor(), ])
    train_data = torchvision.datasets.MNIST(
        root=dataset_path,
        train=True,
        transform=transform,
        download=True,
    )
elif dataset == 'CIFAR10':
    train_data = torchvision.datasets.CIFAR10(
        root=dataset_path,
        train=True,
        transform=transform,
        download=True,
    )
elif dataset == 'CelebA':
    train_data = torchvision.datasets.CelebA(
        root=dataset_path,
        split='train',
        transform=transform,
        download=True,
    )

num_training_img = train_data.__len__()

train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=True,
)

nvae = NvaeModel(num_in_channel=NUM_IN_CHANNEL,
                 num_in_size=NUM_IN_SIZE,
                 batch_size=BATCH_SIZE,
                 latent_z_size=LATENT_Z_SIZE,
                 double_group=DOUBLE_GROUP,)
nvae.cuda()
optimizer = torch.optim.Adam(nvae.parameters(), lr=LR)
loss_hist = []
flag = True

for epoch in range(EPOCH):

    # commands for plot
    num_batch = num_training_img // BATCH_SIZE
    if num_batch <= 400:
        plot_step = 100
    elif num_batch <= 800:
        plot_step = 200
    else:
        plot_step = 300
    plot_set = num_batch // plot_step + 1
    if plot_set <= 2:
        plot_step //= 2
        plot_set *= 2
    fig_size_h = 4 * plot_set * NUM_IN_SIZE[0] / 100
    fig_size_w = 12 * NUM_IN_SIZE[1] / 100
    _, a = plt.subplots(plot_set * 3, 10, figsize=(fig_size_w, fig_size_h), dpi=150)

    for step, (x, b_label) in enumerate(train_loader):
        b_x = x.cuda()

        recon, mu_logvar_list, loss = nvae(b_x)
        if math.isnan(loss):
            flag = False
            break

        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        # commands for plot
        if step % plot_step == 0:
            print('Epoch: ', epoch, ' | step: ', step, ' | train loss: %.4f' % loss.cpu().data.numpy())
            loss_hist.append(loss.cpu().data.numpy())
            img_row = step // plot_step * 3

            for i in range(10):
                if NUM_IN_CHANNEL == 1:
                    a[img_row][i].imshow(np.reshape(b_x.cpu().data[i].numpy(), (32, 32)), cmap='gray')
                    a[img_row + 1][i].imshow(np.reshape(recon.cpu().data[i].numpy(), (32, 32)), cmap='gray')
                else:
                    a[img_row][i].imshow(np.transpose(b_x.cpu().data[i].numpy()/2+0.5, (1, 2, 0)))
                    a[img_row + 1][i].imshow(np.clip(np.transpose(recon.cpu().data[i].numpy() / 2 + 0.5, (1, 2, 0)), 0, 1))
                a[img_row][i].set_xticks(())
                a[img_row][i].set_yticks(())
                a[img_row+1][i].set_xticks(())
                a[img_row+1][i].set_yticks(())
                a[img_row+2][i].set_xticks(())
                a[img_row+2][i].set_yticks(())
    if not flag:
        break
    pic_name = save_pics_path + '/nvae - Epoch ' + str(epoch) + '.png'
    plt.savefig(pic_name, bbox_inches='tight')
    # plt.show()

plt.figure()
plt.plot(loss_hist)
loss_pic_name = save_pics_path + '/nvae - loss - hist.png'
plt.savefig(loss_pic_name, bbox_inches='tight')
# plt.show()

state_name = './states/' + dataset + '.pkl'
torch.save(nvae.state_dict(), state_name)
