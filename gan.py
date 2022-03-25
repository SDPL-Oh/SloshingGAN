import torch
from torch.utils.data import DataLoader
from data_processing import PeakData, PeckDataset, Normalized, ToTensor

import torch.optim as optim
import torchvision.transforms as transforms

latent_size = 16
hidden_size = 256
num_epochs = 10
batch_size = 10
lr = 0.0002
beta1 = 0.5
num_gpu = 2
cond_header = ['Hs', 'Tz', 'Speed', 'Heading', 'sensor']
output_header = ['Peak Pressure(bar)']
device = torch.device("cuda:0" if (torch.cuda.is_available() and num_gpu > 0) else "cpu")

# TODO
#   1. gpu 설정
#   2. 파라미터 변경
#   3. 저장 기능
#   4. 전체 데이터 사용

class Dataset:
    def __init__(self):
        self.csv_file = 'preprocessing/train.csv'

    def load_dataset(self):
        peak_dataset = PeckDataset(csv_file=self.csv_file,
                                   cond_header=cond_header,
                                   output_header=output_header,
                                   transform=transforms.Compose([Normalized(self.csv_file, cond_header),
                                                                 ToTensor()]))

        data_loader = DataLoader(dataset=peak_dataset,
                                 batch_size=batch_size,
                                 shuffle=False)

        real_batch = next(iter(data_loader))
        print("Print sample data: ", real_batch)

        return data_loader


class Generator(torch.nn.Module):
    def __init__(self, num_gpu):
        super(Generator, self).__init__()
        self.num_gpu = num_gpu
        self.main = torch.nn.Sequential(
            torch.nn.Linear(latent_size + len(cond_header), 20),
            torch.nn.BatchNorm1d(20),
            torch.nn.ReLU(True),
            torch.nn.Linear(20, 10),
            torch.nn.BatchNorm1d(10),
            torch.nn.ReLU(True),
            torch.nn.Linear(10, 5),
            torch.nn.BatchNorm1d(5),
            torch.nn.ReLU(True),
            torch.nn.Linear(5, 1)
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(torch.nn.Module):
    def __init__(self, num_gpu):
        super(Discriminator, self).__init__()
        self.num_gpu = num_gpu
        self.main = torch.nn.Sequential(
            torch.nn.Linear(1, 10, bias=False),
            torch.nn.BatchNorm1d(10),
            torch.nn.ReLU(True),
            torch.nn.Linear(10, 10),
            torch.nn.BatchNorm1d(10),
            torch.nn.ReLU(True),
            torch.nn.Linear(10, 5),
            torch.nn.BatchNorm1d(5),
            torch.nn.ReLU(True),
            torch.nn.Linear(5, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


class BuildModel:
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0)

    def apply_init(self):
        gen = Generator(num_gpu).to(device)
        if (device.type == 'cuda') and (num_gpu > 1):
            gen = torch.nn.DataParallel(gen, list(range(num_gpu)))
        gen.apply(self.weights_init)
        dis = Discriminator(num_gpu).to(device)
        if (device.type == 'cuda') and (num_gpu > 1):
            dis = torch.nn.DataParallel(dis, list(range(num_gpu)))
        dis.apply(self.weights_init)

        print("########## Models Summary Start ##########")
        print(gen)
        print(dis)
        print("########## Models Summary End ##########")

        return gen, dis


class Training:
    def __init__(self):
        dataset = Dataset()
        models = BuildModel()
        self.data_loader = dataset.load_dataset()
        self.gen, self.dis = models.apply_init()

    def train(self):
        peak_list = []
        G_losses = []
        D_losses = []
        criterion = torch.nn.BCELoss()
        iters = 0
        real_label = 1.
        fake_label = 0.
        optimizerD = optim.Adam(self.dis.parameters(), lr=lr, betas=(beta1, 0.999))
        optimizerG = optim.Adam(self.gen.parameters(), lr=lr, betas=(beta1, 0.999))

        print("Starting Training Loop...")
        for epoch in range(num_epochs):
            for i, data in enumerate(self.data_loader, 0):
                self.dis.zero_grad()
                real = data['results'].to(device)
                label = torch.full((real.size(0), ), real_label, dtype=torch.float, device=device)
                output = self.dis(real).view(-1)
                errD_real = criterion(output, label)
                errD_real.backward()
                D_x = output.mean().item()

                noise = torch.randn((real.size(0), latent_size), device=device)
                input_cond = torch.cat([data['conditions'].to(device), noise], dim=1)
                fake = self.gen(input_cond)
                label.fill_(fake_label)
                output = self.dis(fake.detach()).view(-1)
                errD_fake = criterion(output, label)
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                errD = errD_real + errD_fake
                optimizerD.step()

                self.gen.zero_grad()
                label.fill_(real_label)
                output = self.dis(fake).view(-1)
                errG = criterion(output, label)
                errG.backward()
                D_G_z2 = output.mean().item()
                optimizerG.step()

                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch, num_epochs, i, len(self.data_loader),
                             errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                G_losses.append(errG.item())
                D_losses.append(errD.item())

                fixed_noise = torch.randn((real.size(0), latent_size), device=device)
                fixed_cond = torch.cat([data['conditions'].to(device), fixed_noise], dim=1)
                if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(self.data_loader) - 1)):
                    with torch.no_grad():
                        fake = self.gen(fixed_cond).detach().cpu()
                    peak_list.append(fake)

                iters += 1

a = Training()
a.train()