"""
Implementation of AlexNet
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

# define pytorch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  define model parameters
NUM_EPOCHS = 90
BATCH_SIZE = 128
MOMENTUM = 0.9
LR_DECAY = 0.0005
LR_INIT = 0.01
IMAGE_DIM = 227
NUM_CLASSES = 1000
DEVICE_IDS = [0,1,2,3]

#  path to data directory
INPUT_ROOT_DIR = 'imagenet'
TRAIN_IMG_DIR = 'imagenet/train'
VAL_IMG_DIR = 'imagenet/val'
OUTPUT_DIR = 'output'
LOG_DIR = OUTPUT_DIR + "/tblogs"
CHECKPOINT_DIR = OUTPUT_DIR + '/models'

# make checkpoint path directory
os.makedirs(LOG_DIR,exist_ok=True)
os.makedirs(CHECKPOINT_DIR,exist_ok=True)

class AlexNet(nn.Module):

    def __init__(self,num_classes = 1000):
        """
               Define and allocate layers for this neural net.

               Args:
                   num_classes (int): number of classes to predict with this model
        """
        super().__init__()
        # input size should be : (b x 3 x 227 x 227)
        # five layers of convolution
        self.net = nn.Sequential(
            #  Layer 1
            nn.Conv2d(in_channels=3,out_channels=96,kernel_size=11,stride=4),  #valid_conv (b x 96 x 55 x 55)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5,alpha=0.0001,beta=0.75,k=2),
            nn.MaxPool2d(kernel_size=3,stride=2),  #max_pool (b x 96 x 27 x 27)
            #  Layer 2
            nn.Conv2d(in_channels=96,out_channels=256,kernel_size=5,padding=2),  #same_conv (b x 256 x 27 x 27)   net[4]
            nn.ReLU(),
            nn.LocalResponseNorm(size=5,alpha=0.0001,beta=0.75,k=2),
            nn.MaxPool2d(kernel_size=3,stride=2),  #max_pool (b x 256 x 13 x 13)
            #  Layer 3
            nn.Conv2d(in_channels=256,out_channels=384,kernel_size=3,padding=1),  #same_conv (b x 384 x 13 x 13)
            nn.ReLU(),
            #  Layer 4
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),  #same_conv (b x 384 x 13 x 13)    net[10]
            nn.ReLU(),
            #  Layer 5
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),  #same_conv (b x 256 x 13 x 13)   net[12]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),  #max_pool (b x 256 x 6 x 6)
        )
        # three layers of fully_connect
        self.classifier = nn.Sequential(
            #  Layer 6
            nn.Dropout(p=0.5,inplace=False),
            nn.Linear(in_features=(256 * 6 * 6),out_features=4096),  #fc (b x 4096 x 1)
            nn.ReLU(),
            #  Layer 7
            nn.Dropout(p=0.5,inplace=False),
            nn.Linear(in_features=4096,out_features=4096),  #fc (b x 4096 x 1)
            nn.ReLU(),
            #  Layer 8
            nn.Linear(in_features=4096,out_features=num_classes),
        )
        # initialize model parameters
        self.init_param()

    def init_param(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight,mean=0,std=0.01)
                nn.init.constant_(layer.bias,val=0)
        # in original paper, Conv2d layers 2nd, 4th and 5th have bias of 1
        nn.init.constant_(self.net[4].bias,val=1),
        nn.init.constant_(self.net[10].bias,val=1),
        nn.init.constant_(self.net[12].bias, val=1),

        for layer in self.classifier:
            if isinstance(layer,nn.Linear):
                nn.init.normal_(layer.weight,mean=0,std=0.01)
                nn.init.constant_(layer.bias,val=1)

    def forward(self,x):
        """
               Pass the input through the alexnet.

               Args:
                   x (Tensor): input tensor

               Returns:
                   output (Tensor): output tensor
        """
        x = self.net(x)
        x = x.view(-1,256 * 6 * 6)  # flatten
        return self.classifier(x)
if __name__ == '__main__':
    # seed
    seed = torch.initial_seed()
    print(f'Used seed:{seed}')

    # TensorboardX summary
    tbwriter = SummaryWriter(log_dir=LOG_DIR)
    print("Tensorborad summary writer created")

    # create model
    alexnet = AlexNet(num_classes=NUM_CLASSES).to(device)
    # train on multiple GPUs
    alexnet = torch.nn.parallel.DataParallel(alexnet,device_ids=DEVICE_IDS)
    print(alexnet)
    print("AlexNet created")

    #transforms
    transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_DIM,scale=(0.9,1.0),ratio=(0.9,1,1)),
        transforms.CenterCrop(IMAGE_DIM),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),  #why?
    ])

    #create dataset and dataloader
    train_dataset = datasets.ImageFolder(TRAIN_IMG_DIR,transform=transform)
    print("Train dataset created")

    val_dataset = datasets.ImageFolder(VAL_IMG_DIR,transform=transform)
    print("Validation dataset created")

    train_dataloader = data.DataLoader(
        train_dataset,
        shuffle=True,
        pin_memory=True,
        num_workers=12,
        drop_last=True,
        batch_size=BATCH_SIZE
    )
    print("Train dataloader created")

    val_dataloader = data.DataLoader(
        val_dataset,
        shuffle=True,
        pin_memory=True,
        num_workers=12,
        drop_last=True,
        batch_size=BATCH_SIZE
    )



    #create optimizer
    # the one that WORKS
    optimizer = optim.Adam(params=alexnet.parameters(), lr=0.0001)
    ### BELOW is the setting proposed by the original paper - which doesn't train....
    # optimizer = optim.SGD(
    #     params=alexnet.parameters(),
    #     lr=LR_INIT, # 0.01
    #     momentum=MOMENTUM, # 0.9
    #     weight_decay=LR_DECAY) # 0.0005
    print('Optimizer created')

    #multiply LR by 1 /10 after every 30 epochs
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=30,gamma=0.1)
    print("LR Scheduler create")

    #start training
    print("Start training...")
    total_steps = 1
    for epoch in range(NUM_EPOCHS):
        for imgs, classes in train_dataloader:
            imgs, classes = imgs.to(device), classes.to(device)

            #calculate the loss
            output = alexnet(imgs)
            loss = F.cross_entropy(output,classes)

            #update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #log the information and add to tensorboard
            if total_steps % 10 == 0:
                with torch.no_grad():
                    _,preds = torch.max(output,1)  # torch.max(...) return (chance,class)
                    accuracy = torch.sum(preds == classes)

                    print(f'Epoch: {epoch + 1} \tStep: {total_steps} \tLoss: {loss.item()} \tAcc: {accuracy.item()}')
                    tbwriter.add_scalar('loss',loss.item(),total_steps)
                    tbwriter.add_scalar("accuracy",accuracy.item(),total_steps)

            #print out gradient values and parameter average values
            if total_steps%1000 == 0:
                with torch.no_grad():
                    # print and save the grad of the parameters
                    # also print and save parameter valuse
                    print('*' * 10)
                    for name, parameter in alexnet.named_parameters():
                        if parameter.grad is not None:
                            avg_grad = torch.mean(parameter.grad)
                            print(f'\t{name} - grad_avg: {avg_grad}')
                            tbwriter.add_scalar(f'grad_avg/{name}',avg_grad.item(),total_steps)
                            tbwriter.add_histogram(f'grad/{name}',parameter.grad.cpu().numpy(),total_steps)

                        if parameter.data is not None:
                            avg_weight = torch.mean(parameter.data)
                            print(f'\t{name} - param_avg: {avg_weight}')
                            tbwriter.add_histogram(f'weight/{name}',parameter.data.cpu().numpy(), total_steps)
                            tbwriter.add_scalar(f'weight_avg/{name}', avg_weight.item(), total_steps)

            total_steps += 1
        lr_scheduler.step()

        # save checkpoints (every epoch)
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'alexnet_states_e{epoch + 1}.pkl')
        state = {
            'epoch': epoch,
            "total_steps": total_steps,
            "optimizer": optimizer.state_dict(),
            'model': alexnet.state_dict(),
            'seed': seed,
        }
        torch.save(state,checkpoint_path)





