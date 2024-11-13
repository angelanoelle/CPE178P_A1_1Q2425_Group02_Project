import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

"""ResNet."""
# Define the weight initialization function.
def _weight_variable(shape, factor=0.01):
    init_value = np.random.randn(*shape).astype(np.float32) * factor
    return Tensor(init_value)

# Define the 3x3 convolution layer functions.
def _conv3x3(in_channel, out_channel, stride=1):
    weight_shape = (out_channel, in_channel, 3, 3)
    weight = _weight_variable(weight_shape)
    return nn.Conv2d(in_channel, out_channel,
                    kernel_size=3, stride=stride, padding=0, pad_mode='same', weight_init=weight)

# Define the 1x1 convolution layer functions.
def _conv1x1(in_channel, out_channel, stride=1):
    weight_shape = (out_channel, in_channel, 1, 1)
    weight = _weight_variable(weight_shape)
    return nn.Conv2d(in_channel, out_channel,
                    kernel_size=1, stride=stride, padding=0, pad_mode='same', weight_init=weight)

# Define the 7x7 convolution layer functions.
def _conv7x7(in_channel, out_channel, stride=1):
    weight_shape = (out_channel, in_channel, 7, 7)
    weight = _weight_variable(weight_shape)
    return nn.Conv2d(in_channel, out_channel,
                    kernel_size=7, stride=stride, padding=0, pad_mode='same', weight_init=weight)

# Define the Batch Norm layer functions.
def _bn(channel):
    return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.9,
                        gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)

# Define the Batch Norm functions at the last layer.
def _bn_last(channel):
    return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.9,
                        gamma_init=0, beta_init=0, moving_mean_init=0, moving_var_init=1)

# Define the functions of the fully-connected layers.
def _fc(in_channel, out_channel):
    weight_shape = (out_channel, in_channel)
    weight = _weight_variable(weight_shape)
    return nn.Dense(in_channel, out_channel, has_bias=True, weight_init=weight, bias_init=0)

# Construct a residual module.
class ResidualBlock(nn.Cell):
    """
        ResNet V1 residual block definition.
        Args:
        in_channel (int): Input channel.
        out_channel (int): Output channel.
        stride (int): Stride size for the first convolutional layer. Default: 1.
        Returns:
        Tensor, output tensor.
        Examples:
        >>> ResidualBlock(3, 256, stride=2)
    """
    expansion = 4 # In conv2_x--conv5_x, the number of convolution kernels at the first two layers is one fourth of the number of convolution kernels at the third layer (an output channel).

    def __init__(self, in_channel, out_channel, stride=1):
        super(ResidualBlock, self).__init__()
        # The number of convolution kernels at the first two layers is equal to a quarter of the number of convolution kernels at the output channels.
        channel = out_channel // self.expansion

        # Layer 1 convolution
        self.conv1 = _conv1x1(in_channel, channel, stride=1)
        self.bn1 = _bn(channel)

        # Layer 2 convolution
        self.conv2 = _conv3x3(channel, channel, stride=stride)
        self.bn2 = _bn(channel)

        # Layer 3 convolution. The number of convolution kernels is equal to that of output channels.
        self.conv3 = _conv1x1(channel, out_channel, stride=1)
        self.bn3 = _bn_last(out_channel)

        # ReLU activation layer
        self.relu = nn.ReLU()
        self.down_sample = False

        # When the step is not 1 or the number of output channels is not equal to that of input channels, downsampling is performed to adjust the number of channels.
        if stride != 1 or in_channel != out_channel:
            self.down_sample = True
        self.down_sample_layer = None

        # Adjust the number of channels using the 1x1 convolution.
        if self.down_sample:
            self.down_sample_layer = nn.SequentialCell([_conv1x1(in_channel, out_channel, stride), #1x1
                                    _bn(out_channel)]) # Batch Norm

        # Addition operator
        self.add = ops.Add()

    # Construct a residual block.
    def construct(self, x):
        # Input
        identity = x

        # Layer 1 convolution 1x1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Layer 2 convolution 3x3
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # Layer 3 convolution 1x1
        out = self.conv3(out)
        out = self.bn3(out)

        # Change the network dimension.
        if self.down_sample:
            identity = self.down_sample_layer(identity)

        # Add the residual.
        out = self.add(out, identity)

        # ReLU activation
        out = self.relu(out)

        return out


# Construct a residual network.
class ResNet(nn.Cell):
    """
        ResNet architecture.
        Args:
        block (Cell): Block for network.
        layer_nums (list): Numbers of block in different layers.
        in_channels (list): Input channel in each layer.
        out_channels (list): Output channel in each layer.
        strides (list): Stride size in each layer.
        num_classes (int): The number of classes that the training images belong to.
        Returns:
        Tensor, output tensor.
        Examples:
        >>> ResNet(ResidualBlock,
        >>> [3, 4, 6, 3],
        >>> [64, 256, 512, 1024],
        >>> [256, 512, 1024, 2048],
        >>> [1, 2, 2, 2],
        >>> 10)
    """

    # Input parameters: residual block, number of repeated residual blocks, input channel, output channel, stride,and number of image classes

    def __init__(self, block, layer_nums, in_channels, out_channels, strides, num_classes):
        super(ResNet, self).__init__()
        if not len(layer_nums) == len(in_channels) == len(out_channels) == 4:
            raise ValueError("the length of layer_num, in_channels, out_channels list must be 4!")

        # Layer 1 convolution; convolution kernels: 7x7, input channels: 3; output channels: 64; step: 2
        self.conv1 = _conv7x7(3, 64, stride=2)
        self.bn1 = _bn(64)
        self.relu = ops.ReLU()

        # 3x3 pooling layer; step: 2
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")

        # conv2_x residual block
        self.layer1 = self._make_layer(block,
                                       layer_nums[0],
                                       in_channel=in_channels[0],
                                       out_channel=out_channels[0],
                                       stride=strides[0])

        # conv3_x residual block
        self.layer2 = self._make_layer(block,
                                       layer_nums[1],
                                       in_channel=in_channels[1],
                                       out_channel=out_channels[1],
                                       stride=strides[1])

        # conv4_x residual block
        self.layer3 = self._make_layer(block,
                                       layer_nums[2],
                                       in_channel=in_channels[2],
                                       out_channel=out_channels[2],
                                       stride=strides[2])

        # conv5_x residual block
        self.layer4 = self._make_layer(block, layer_nums[3],
                                       in_channel=in_channels[3],
                                       out_channel=out_channels[3],
                                       stride=strides[3])

        # Mean operator
        self.mean = ops.ReduceMean(keep_dims=True)

        # Flatten layer
        self.flatten = nn.Flatten()

        # Output layer
        self.end_point = _fc(out_channels[3], num_classes)

    # Input parameters: residual block, number of repeated residual blocks, input channel, output channel, and stride
    def _make_layer(self, block, layer_num, in_channel, out_channel, stride):
        """
            Make stage network of ResNet.
            Args:
            block (Cell): Resnet block.
            layer_num (int): Layer number.
            in_channel (int): Input channel.
            out_channel (int): Output channel.
            stride (int): Stride size for the first convolutional layer.
            Returns:
            SequentialCell, the output layer.
            Examples:
            >>> _make_layer(ResidualBlock, 3, 128, 256, 2)
        """

        # Build the residual block of convn_x.
        layers = []
        resnet_block = block(in_channel, out_channel, stride=stride)
        layers.append(resnet_block)

        for _ in range(1, layer_num):
            resnet_block = block(out_channel, out_channel, stride=1)
            layers.append(resnet_block)

        return nn.SequentialCell(layers)

    # Build a ResNet network.
    def construct(self, x):
        x = self.conv1(x) # Layer 1 convolution: 7x7; step: 2
        x = self.bn1(x) # Batch Norm of layer 1
        x = self.relu(x) # ReLU activation layer
        c1 = self.maxpool(x) # Max pooling: 3x3; step: 2

        c2 = self.layer1(c1) # conv2_x residual block
        c3 = self.layer2(c2) # conv3_x residual block
        c4 = self.layer3(c3) # conv4_x residual block
        c5 = self.layer4(c4) # conv5_x residual block

        out = self.mean(c5, (2, 3)) # Mean pooling layer
        out = self.flatten(out) # Flatten layer
        out = self.end_point(out) # Output layer

        return out

# Build a ResNet-50 network.
def resnet50(class_num=5):
    """
        Get ResNet50 neural network.
        Args:
        class_num (int): Class number.
        Returns:
        Cell, cell instance of ResNet50 neural network.
        Examples:
        >>> net = resnet50(10)
    """
    return ResNet(ResidualBlock, # Residual block
        [3, 4, 6, 3], # Number of residual blocks
        [64, 256, 512, 1024], # Input channel
        [256, 512, 1024, 2048], # Output channel
        [1, 2, 2, 2], # Step
        class_num) # Number of output classes
