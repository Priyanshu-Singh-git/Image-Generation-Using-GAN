# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 11:26:08 2024

@author: Priyanshu singh
"""
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.utils as vutils
import torch.optim as optim
import numpy as np
from torch.utils.data import Subset

import seaborn as sns

# Variables
image_size = 64
batch_size = 128
ngpu =1
workers = 0
ngf = 64
nc = 3
nz = 100
ndf = 64
lr = 0.0002
beta1 = 0.5
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
dataroot = "D:/pythonProject/GAN celeb A/celeba"
dataset_size = 200000
indices = list(range(dataset_size))

# PREPARING DATASET

dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

dataset1 = Subset(dataset, indices)
dataloader = torch.utils.data.DataLoader(dataset1, batch_size=batch_size,shuffle=True, num_workers=workers)

#%%
# FUNCTION TO INITIALIZE THE WEIGHTS OF THE GAN NEURAL NETWORKS

def weights_init(m):
    
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
       


class Generator(nn.Module):
    def __init__(self,ngpu):
        super(Generator,self).__init__()
        self.ngpu = ngpu
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        self.main = nn.Sequential(
            
            nn.ConvTranspose2d(self.nz, self.ngf*8,4,1,0,bias=False),
            nn.BatchNorm2d(self.ngf*8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.ngf*8, self.ngf*4,4,2,1,bias=False),
            nn.BatchNorm2d(self.ngf*4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.ngf*4, self.ngf*2,4,2,1,bias=False),
            nn.BatchNorm2d(self.ngf*2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.ngf*2, self.ngf,4,2,1,bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.ngf, self.nc, 4,2,1,bias=False),
            nn.Tanh()
            )

    def forward(self,input):
        return self.main(input)
    


class Discriminator(nn.Module):
    def __init__(self,ngpu):
        self.nc = nc
        self.ngpu = ngpu
        super(Discriminator,self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(self.nc, ndf, 4,2,1,bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            
            nn.Conv2d(ndf, ndf*2,4,2,1,bias = False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2,inplace = True),
            
            nn.Conv2d(ndf*2, ndf*4,4,2,1,bias = False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2,inplace = True),
            
            nn.Conv2d(ndf*4, ndf*8,4,2,1,bias = False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2,inplace = True),
            
            nn.Conv2d(ndf*8, 1,4,1,0,bias = False),
            
            nn.Sigmoid(),
            
            )
    def forward(self,input):
        return self.main(input)
    
netg = Generator(ngpu).to(device)

if (device.type == 'cuda') and (ngpu > 1):
    netg = nn.DataParallel(netg, list(range(ngpu)))

netg.apply(weights_init)

print(netg)
netd = Discriminator(ngpu).to(device)

if (device.type == 'cuda') and (ngpu > 1):
    netd= nn.DataParallel(netd, list(range(ngpu)))

netd.apply(weights_init)

print(netd)


criterion = nn.BCELoss()
optimizerD = optim.Adam(netd.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netg.parameters(), lr=lr, betas=(beta1, 0.999))

real_label = 1
fake_label = 0
fixed_noise = torch.randn(64,nz,1,1,device=device)


img_list = []
G_losses = []
D_losses = []
iters = 0
num_epochs = 8

print("Starting Training Loop...")
#%%

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        print('training started')
        
        netd.zero_grad()
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

        output = netd(real_cpu).view(-1)
    
        errD_real = criterion(output, label)
     
        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(b_size, nz, 1, 1, device=device)
 
        fake = netg(noise)
        label.fill_(fake_label)
       
        output = netd(fake.detach()).view(-1)
       
        errD_fake = criterion(output, label)
      
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        
        errD = errD_real + errD_fake
       
        optimizerD.step()

       
        netg.zero_grad()
        label.fill_(real_label)  
        output = netd(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()
        
        
        if i%50==0:
            # probabiliy distribution visualization
            with torch.no_grad():
                fake_data = netg(noise)
                fake_data = fake_data.cpu().numpy().flatten()
                real_data = real_cpu.cpu().numpy().flatten()
            plt.figure(figsize=(12, 6))
            sns.kdeplot(real_data, label='Real Data', fill=True, color='blue')
            sns.kdeplot(fake_data, label='Generated Data', fill=True, color='red')
        
            plt.title('Probability Distribution of Real and Generated Data')
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.legend()
            plt.show()
            
            
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        G_losses.append(errG.item())
        D_losses.append(errD.item())

        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netg(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            
            
       
        
            
            
        iters += 1
        
#%%
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()


import pickle


with open('D:/pythonProject/GAN celeb A/my_list.pkl', 'wb') as file:
    pickle.dump(img_list, file)

PATH = "D:/pythonProject/GAN celeb A/gener.pth"
Path2 = "D:/pythonProject/GAN celeb A/discr.pth"

# Save
torch.save(netg.state_dict(), PATH)
torch.save(netd.state_dict(),Path2)
'''
import matplotlib.animation as animation
from IPython.display import HTML
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())'''
#%%
netg = Generator(ngpu).to(device)
weights = torch.load("D:/pythonProject/GAN celeb A/gener.pth")
netg.load_state_dict(weights)

noise = torch.randn(1, 100, 1, 1, device=device)
fake = netg(noise).detach().cpu()
fake = fake.squeeze(0)

print(fake.size())
fake = fake.permute(1,2,0)

img_np= np.array(fake).astype(np.float32)
img_np = ((img_np+1)/2)*255
img_np = img_np.astype(np.uint8)
plt.imshow(img_np)
plt.show()







