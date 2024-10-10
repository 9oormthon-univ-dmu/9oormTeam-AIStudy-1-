

import torch
from torch import nn 


class CNNAugment_TinyVGG(nn.Module):

    
    def __init__(self, num_channels, num_filters, num_classes): 
        
        super().__init__()
        
        
        self.conv_block_entrance = nn.Sequential(
            
            nn.Conv2d(in_channels=num_channels, # will be '3' == R/G/B
                      out_channels=num_filters, # num_filters == num of feature-maps == num of output channels
                      kernel_size=(3, 3), 
                      stride=1,                 # default
                      padding=1),               # 0 == 'valid', 1 == 'same'
            nn.BatchNorm2d(num_filters),        # Batch-normalization on 2-Dimensional data
            nn.ReLU(),
            
            nn.Conv2d(in_channels=num_filters,  # should be same as the number of "channels of previous output"
                      out_channels=num_filters,
                      kernel_size=(3, 3),
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(num_filters),        # Batch-normalization 
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2, stride=2) # Default == kernel_size (자동으로 지정됨)
        )
        # [ 32, 3, 64, 64 ] -> [ 32, 10, 32, 32 ]
        
        
        self.conv_block_hidden_1 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, (3, 3), padding=1),
            nn.BatchNorm2d(num_filters), # Batch-normalization 
            nn.ReLU(),
            nn.Conv2d(num_filters, num_filters, (3, 3), padding=1),
            nn.BatchNorm2d(num_filters), # Batch-normalization 
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # [ 32, 10, 32, 32 ] -> [ 32, 10, 16, 16 ]
        
        
        self.conv_block_hidden_2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, (3, 3), padding=1),
            nn.BatchNorm2d(num_filters), # Batch-normalization 
            nn.ReLU(),
            nn.Conv2d(num_filters, num_filters, (3, 3), padding=1),
            nn.BatchNorm2d(num_filters), # Batch-normalization 
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # [ 32, 10, 16, 16 ] -> [ 32, 10, 8, 8 ]
        
        
        self.classifier_block = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5), # Drop-out
            nn.Linear(in_features=num_filters * 8 * 8, 
                      out_features=num_filters * 16 * 16),
            nn.ReLU(),
            nn.Dropout(0.5), # Drop-out
            nn.Linear(in_features=num_filters * 16 * 16, 
                      out_features=num_classes)
        )
        # [ 32, 10, 8, 8 ] -> [ 32, 10 * 8 * 8 ] -> [ 32, 10 * 16 * 16 ] -> [ 32, 10 ]
    
    
    def forward(self, x):
        
        x = self.conv_block_entrance(x)
        # print(x.shape)
        
        x = self.conv_block_hidden_1(x)
        # print(x.shape)
        
        x = self.conv_block_hidden_2(x)
        # print(x.shape)
        
        x = self.classifier_block(x)
        # print(x.shape)
        
        # 아래와 같이 코드를 작성하게되면 메모리가 크게 소요되는 변수 재할당 과정이 생략되므로 계산 속도가 향상됩니다. (https://bit.ly/3V16ZJy)
        # return self.classifier_block(conv_block_hidden_2(conv_block_hidden_1(conv_block_entrance(x)))
        
        return x
