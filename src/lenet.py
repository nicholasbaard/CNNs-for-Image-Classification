import torch.nn as nn
import torch
from torchinfo import summary


class LeNet5(nn.Module):
    def __init__(self, input_dims=[1,32,32], num_classes=10):
        super(LeNet5, self).__init__()

        self.feature_extraction = nn.Sequential(
            nn.Conv2d(input_dims[0], 6, kernel_size=5), #C1 Output: [6, 28, 28]
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2), #S2 Output: [6, 14, 14]
            
            nn.Conv2d(6, 16, kernel_size=5), #C3 Output: [16, 10, 10]
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2), #S4 Output: [16, 5, 5]
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16*5*5, 84), #C5 Output: [84]
            nn.Tanh(),
            nn.Linear(84, num_classes), #F6 Output: [10]
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.fc(x)
        return x
        

if __name__ == "__main__":

    lenet = LeNet5()

    summ =  summary(model=lenet, input_size=(1, 1, 32, 32), col_width=20,
                    col_names=['input_size', 'output_size', 'num_params', 'trainable'], row_settings=['var_names'], verbose=0)
    
    print(summ)