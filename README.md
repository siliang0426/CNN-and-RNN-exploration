# CNN-and-RNN-exploration
### This is project is under the instruction of Professor Adriana Kovashka and it was designed by Mingda Zhang
### The purpose of this project is to reconstruct the Convolution Neural Network such as the AlexNET and the variation of it and rebuilding the classical Recurrance Nerual Network of GRU, LSTM, Peephole LSTM, and Coupled LSTM.
#### The reference of the RNN could be found here: https://colah.github.io/posts/2015-08-Understanding-LSTMs/
### Below is the structure of the AlexNET and it variation
## AlexNET
    Layer (type)      Kernel      Padding      Stride      Dilation          Output Shape           Param #     
----------------------------------------------------------------------------------------------------------------
    Layer (type)      Kernel      Padding      Stride      Dilation          Output Shape           Param #   

        Conv2d-1     11 x 11                        4                    [-1, 96, 55, 55]            34,944
          ReLU-2                                                         [-1, 96, 55, 55]                 0
     MaxPool2d-3           3                        2                    [-1, 96, 27, 27]                 0
        Conv2d-4       5 x 5            2                               [-1, 256, 27, 27]           614,656
          ReLU-5                                                        [-1, 256, 27, 27]                 0
     MaxPool2d-6           3                        2                   [-1, 256, 13, 13]                 0
        Conv2d-7       3 x 3            1                               [-1, 384, 13, 13]           885,120
          ReLU-8                                                        [-1, 384, 13, 13]                 0
        Conv2d-9       3 x 3            1                               [-1, 384, 13, 13]         1,327,488
         ReLU-10                                                        [-1, 384, 13, 13]                 0
       Conv2d-11       3 x 3            1                               [-1, 256, 13, 13]           884,992
         ReLU-12                                                        [-1, 256, 13, 13]                 0
    MaxPool2d-13           3                        2                     [-1, 256, 6, 6]                 0
      Flatten-14                                                               [-1, 9216]                 0
      Dropout-15                                                               [-1, 9216]                 0
       Linear-16                                                               [-1, 4096]        37,752,832
         ReLU-17                                                               [-1, 4096]                 0
      Dropout-18                                                               [-1, 4096]                 0
       Linear-19                                                               [-1, 4096]        16,781,312
         ReLU-20                                                               [-1, 4096]                 0
       Linear-21                                                                  [-1, 4]            16,388

## AlexNET Large Kernel
----------------------------------------------------------------------------------------------------------------
    Layer (type)      Kernel      Padding      Stride      Dilation          Output Shape           Param #  
        Conv2d-1     21 x 21            1           8                    [-1, 96, 27, 27]           127,104
          ReLU-2                                                         [-1, 96, 27, 27]                 0
        Conv2d-3       7 x 7            2           2                   [-1, 256, 13, 13]         1,204,480
          ReLU-4                                                        [-1, 256, 13, 13]                 0
        Conv2d-5       3 x 3            1                               [-1, 384, 13, 13]           885,120
          ReLU-6                                                        [-1, 384, 13, 13]                 0
        Conv2d-7       3 x 3            1                               [-1, 384, 13, 13]         1,327,488
          ReLU-8                                                        [-1, 384, 13, 13]                 0
        Conv2d-9       3 x 3                        2                     [-1, 256, 6, 6]           884,992
         ReLU-10                                                          [-1, 256, 6, 6]                 0
      Flatten-11                                                               [-1, 9216]                 0
      Dropout-12                                                               [-1, 9216]                 0
       Linear-13                                                               [-1, 4096]        37,752,832
         ReLU-14                                                               [-1, 4096]                 0
      Dropout-15                                                               [-1, 4096]                 0
       Linear-16                                                               [-1, 4096]        16,781,312
         ReLU-17                                                               [-1, 4096]                 0
       Linear-18                                                                  [-1, 4]            16,388

## AlexNET Average Pooling

----------------------------------------------------------------------------------------------------------------
    Layer (type)      Kernel      Padding      Stride      Dilation          Output Shape           Param #
        Conv2d-1     11 x 11                        4                    [-1, 96, 55, 55]            34,944
          ReLU-2                                                         [-1, 96, 55, 55]                 0
     AvgPool2d-3           3                        2                    [-1, 96, 27, 27]                 0
        Conv2d-4       5 x 5            2                               [-1, 256, 27, 27]           614,656
          ReLU-5                                                        [-1, 256, 27, 27]                 0
     AvgPool2d-6           3                        2                   [-1, 256, 13, 13]                 0
        Conv2d-7       3 x 3            1                               [-1, 384, 13, 13]           885,120
          ReLU-8                                                        [-1, 384, 13, 13]                 0
        Conv2d-9       3 x 3            1                               [-1, 384, 13, 13]         1,327,488
         ReLU-10                                                        [-1, 384, 13, 13]                 0
       Conv2d-11       3 x 3            1                               [-1, 256, 13, 13]           884,992
         ReLU-12                                                        [-1, 256, 13, 13]                 0
    AvgPool2d-13           3                        2                     [-1, 256, 6, 6]                 0
      Flatten-14                                                               [-1, 9216]                 0
      Dropout-15                                                               [-1, 9216]                 0
       Linear-16                                                               [-1, 4096]        37,752,832
         ReLU-17                                                               [-1, 4096]                 0
      Dropout-18                                                               [-1, 4096]                 0
       Linear-19                                                               [-1, 4096]        16,781,312
         ReLU-20                                                               [-1, 4096]                 0
       Linear-21                                                                  [-1, 4]            16,388