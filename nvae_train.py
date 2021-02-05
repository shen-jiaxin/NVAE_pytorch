import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import platform
from nvae import NvaeModel


EPOCH = 100
BATCH_SIZE = 256
LR = 0.0001
NUM_WORKERS = 0 if platform.system() == 'Windows' else 4
datasets = ['MNIST', 'CIFAR10', 'CelebA']
dataset = datasets[2]
NUM_IN_SIZE = 32
NUM_IN_CHANNEL = 3

save_pics_path = './pics/' + dataset
dataset_path = 'D:\Data\SoftwareSave\Python\Datasets'


transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize((NUM_IN_SIZE, NUM_IN_SIZE)),
     torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

if dataset == 'MNIST':
    NUM_IN_CHANNEL = 1
    num_training_img = 60000
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((NUM_IN_SIZE, NUM_IN_SIZE)),
         torchvision.transforms.ToTensor(), ])
    train_data = torchvision.datasets.MNIST(
        root=dataset_path,
        train=True,
        transform=transform,
        download=True,
    )
elif dataset == 'CIFAR10':
    num_training_img = 50000
    train_data = torchvision.datasets.CIFAR10(
        root=dataset_path,
        train=True,
        transform=transform,
        download=True,
    )
elif dataset == 'CelebA':
    num_training_img = 162770
    train_data = torchvision.datasets.CelebA(
        root=dataset_path,
        split='train',
        transform=transform,
        download=True,
    )

train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=True,
)

recon_loss_func = nn.MSELoss(reduction='sum').cuda()

nvae = NvaeModel(num_in_channel=NUM_IN_CHANNEL, num_in_size=NUM_IN_SIZE, batch_size=BATCH_SIZE)
nvae.cuda()
optimizer = torch.optim.Adam(nvae.parameters(), lr=LR)
loss_hist = []

for epoch in range(EPOCH):
    num_batch = num_training_img // BATCH_SIZE
    if num_batch <= 400:
        plot_step = 100
    elif num_batch <= 800:
        plot_step = 200
    else:
        plot_step = 300
    plot_row = num_batch // plot_step + 1
    if plot_row <= 2:
        plot_step //= 2
        plot_row *= 2
    fig_size = -(-12 * NUM_IN_SIZE * plot_row // 256)

    _, a = plt.subplots(plot_row * 3, 10, figsize=(fig_size, fig_size))

    for step, (x, b_label) in enumerate(train_loader):
        b_x = x.cuda()

        recon_x, loss_kl = nvae(b_x)
        loss = 100 * recon_loss_func(recon_x, b_x) + loss_kl

        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        if step % plot_step == 0:
            print('Epoch: ', epoch, ' | step: ', step, ' | train loss: %.4f' % loss.cpu().data.numpy())
            loss_hist.append(loss.cpu().data.numpy())
            img_row = step // plot_step * 3

            for i in range(10):
                if NUM_IN_CHANNEL == 1:
                    a[img_row][i].imshow(np.reshape(b_x.cpu().data[i].numpy(), (32, 32)), cmap='gray')
                    a[img_row + 1][i].imshow(np.reshape(recon_x.cpu().data[i].numpy(), (32, 32)), cmap='gray')
                else:
                    a[img_row][i].imshow(np.transpose(b_x.cpu().data[i].numpy()/2+0.5, (1, 2, 0)))
                    a[img_row + 1][i].imshow(np.clip(np.transpose(recon_x.cpu().data[i].numpy() / 2 + 0.5, (1, 2, 0)), 0, 1))
                a[img_row][i].set_xticks(())
                a[img_row][i].set_yticks(())
                a[img_row+1][i].set_xticks(())
                a[img_row+1][i].set_yticks(())
                a[img_row+2][i].set_xticks(())
                a[img_row+2][i].set_yticks(())
    pic_name = save_pics_path + '/nvae - Epoch ' + str(epoch) + '.png'
    plt.savefig(pic_name, bbox_inches='tight')
    plt.show()

plt.plot(loss_hist)
loss_pic_name = save_pics_path + '/nvae - loss - hist.png'
plt.savefig(loss_pic_name, bbox_inches='tight')
plt.show()
