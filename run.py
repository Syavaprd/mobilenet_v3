import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

from utils import *

from os import environ
from os.path import abspath, dirname, join

import glob
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
batch_size = 64
EPOCHS = 5

class SqueezeExcite(nn.Module):
    def __init__(self, exp_size, divide=4):
        super(SqueezeExcite, self).__init__()
        
        self.fc_R = nn.Sequential(
            nn.Linear(exp_size, exp_size // divide, bias=False),
            nn.ReLU()
        )
        
        self.fc_HS = nn.Sequential(
            nn.Linear(exp_size // divide, exp_size, bias=False),
            HardSigmoid()
        )

    def forward(self, x):
        batch, channels, height, width = x.size()
        out = F.avg_pool2d(x, kernel_size=[height, width]).view(batch, -1)
        
        out = self.fc_R(out)
        
        out = self.fc_HS(out)

        out = out.view(batch, channels, 1, 1)

        return out * x


class BottleNeck(nn.Module):
    def __init__(self, inp, out, exp, kernal_size, NL, SE, stride):
        super(BottleNeck, self).__init__()
        self.SE = SE
        
        padding = 0
        if kernal_size == 3:
            padding = 1
        elif kernal_size == 5:
            padding = 2
        else:
            assert 0 == 1

        if NL == "ReLu":
            activation = nn.ReLU
        elif NL == "h_swish":
            activation = HardSwish
        else:
            assert 0 == 1

        self.conv1 = nn.Sequential(
            nn.Conv2d(inp, exp, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(exp),
            activation()
        )

        self.dconv1 = nn.Sequential(
            nn.Conv2d(exp, exp, kernel_size=kernal_size, stride=stride, padding=padding, groups=exp, bias=False),
            nn.BatchNorm2d(exp),
        )

        self.squeeze = SqueezeExcite(exp)

        self.conv2 = nn.Sequential(
            nn.Conv2d(exp, out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out),
            activation()
        )
        
        self.connect_flag = (stride == 1 and inp == out)

    def forward(self, x):
        x2 = self.conv1(x)
        x2 = self.dconv1(x2)

        if self.SE:
            x2 = self.squeeze(x2)

        x2 = self.conv2(x2)

        if self.connect_flag:
            return x + x2
        else:
            return x2


class mobilenet_v3(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.0, large=False):
        super(mobilenet_v3, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(16),
                HardSwish(),
        )
        
        self.body = []
        
        if large == True:
            print("Large model\n")
            layers = [
                [16, 16, 16, 3, "ReLu", False, 1],
                [16, 24, 64, 3, "ReLu", False, 2],
                [24, 24, 72, 3, "ReLu", False, 1],
                [24, 40, 72, 5, "ReLu", True, 2],
                [40, 40, 120, 5, "ReLu", True, 1],
                [40, 40, 120, 5, "ReLu", True, 1],
                [40, 80, 240, 3, "h_swish", False, 2],
                [80, 80, 200, 3, "h_swish", False, 1],
                [80, 80, 184, 3, "h_swish", False, 1],
                [80, 80, 184, 3, "h_swish", False, 1],
                [80, 112, 480, 3, "h_swish", True, 1],
                [112, 112, 672, 3, "h_swish", True, 1],
                [112, 160, 672, 5, "h_swish", True, 1],
                [160, 160, 960, 5, "h_swish", True, 2],
                [160, 160, 960, 5, "h_swish", True, 1],
            ]

            for inp, out, exp, kernal_size, NL, SE, stride in layers:
                self.body.append(BottleNeck(inp, out, exp, kernal_size, NL, SE, stride))

            self.pconv2 = nn.Sequential(
                nn.Conv2d(160, 960, kernel_size=1, stride=1),
                nn.BatchNorm2d(960),
                HardSwish(),
            )

            self.pconv3 = nn.Sequential(
                nn.Conv2d(960, 1280, kernel_size=1, stride=1),
                HardSwish(),
                nn.Dropout(p=dropout_rate),
                nn.Conv2d(1280, self.num_classes, kernel_size=1, stride=1),
            )

        else:
            print("Small model\n")
            layers = [
                [16, 16, 16, 3, "ReLu", True, 2],
                [16, 24, 72, 3, "ReLu", False, 2],
                [24, 24, 88, 3, "ReLu", False, 1],
                [24, 40, 96, 5, "ReLu", True, 2],
                [40, 40, 240, 5, "ReLu", True, 1],
                [40, 40, 240, 5, "ReLu", True, 1],
                [40, 48, 120, 5, "h_swish", True, 1],
                [48, 48, 144, 5, "h_swish", True, 1],
                [48, 96, 288, 5, "h_swish", True, 2],
                [96, 96, 576, 5, "h_swish", True, 1],
                [96, 96, 576, 5, "h_swish", True, 1],
            ]

            for inp, out, exp, kernal_size, NL, SE, stride in layers:
                self.body.append(BottleNeck(inp, out, exp, kernal_size, NL, SE, stride))

            self.pconv2 = nn.Sequential(
                nn.Conv2d(96, 576, kernel_size=1, stride=1),
                SqueezeExcite(576),
                nn.BatchNorm2d(576),
                HardSwish(),
            )

            self.pconv3 = nn.Sequential(
                nn.Conv2d(576, 1024, kernel_size=1, stride=1),
                HardSwish(),
                nn.Dropout(p=dropout_rate),
                nn.Conv2d(1024, self.num_classes, kernel_size=1, stride=1),
            )
        
        self.body = nn.Sequential(*self.body)

    def forward(self, x):
        x = self.conv1(x)

        x = self.body(x)

        x = self.pconv2(x)
        
        batch, _, h, w = x.size()
        x = F.avg_pool2d(x, kernel_size=[h, w])
        
        x = self.pconv3(x)

        x = x.view(batch, -1)
        return x

def predict(path, model):
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    outputs = []
    with torch.no_grad():
        for im_path in glob.glob(join(path, '*.jpg')):
            image = Image.open(im_path)
            if image.mode != "RGB":
                continue
                
            image_tensor = test_transforms(image).float()
            
            image_tensor = image_tensor.unsqueeze_(0)
            image_tensor = image_tensor.to(device)

            input = Variable(image_tensor)
            output = model(input)

            index = output.cpu().data.numpy().argmax()
            #print(index)
            outputs.append(index)
            
    return outputs

def train(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 3, gamma = 0.03)

    train_transform = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                           std = [0.229, 0.224, 0.225])
                                      ])

    train_set = torchvision.datasets.CIFAR10(root = './data', train = True,
 	                              download = True, transform = train_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, 
    	                                        shuffle = True)
    train_loss_list = []
    train_acc_list = []

    for epoch in range(EPOCHS):
        print("Epoch: ", epoch + 1)
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs, labels = Variable(inputs), Variable(labels)
    
            optimizer.zero_grad()
    
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
    
        running_loss /= len(train_loader)
        train_loss_list.append(running_loss)
        print('[%d] loss: %.3f' % (epoch + 1, running_loss))
    
        model.eval()
        train_total = 0
        train_correct = 0
    
 
        with torch.no_grad():
            for data in train_loader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
    
                outputs = mobilenet(Variable(images))
                i, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

        train_acc_list.append(100 * train_correct / train_total) 
        print('Train Accuracy: %.3f %%' % (100 * train_correct / train_total))
        scheduler.step()

    return model


if __name__ == '__main__':
    if len(argv) != 4:
        assert 0 == 1
    pretrained = argv[1]
    mode = argv[2]
    path = argv[3]
    if pretrained == "T":
        large = True if mode == "L" else False
        model = mobilenet_v3(large=large)
        if large == True:
            model.load_state_dict(torch.load('./checkpointL.pth'))
        else:
            model.load_state_dict(torch.load('./checkpointS.pth'))
        model.eval()
        model.cuda()

        predictions = predict(path, model)

        print(predictions)
    elif pretrained == "F":
        large = True if mode == "L" else False
        model = mobilenet_v3(large=large)
        model.cuda()

        model = train(model)
        
        model.eval()
        
        print(predict(path, model))
    else:
        assert 0 == 1

	
	
