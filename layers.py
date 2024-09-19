
class Convolution(nn.Module):
    """
    Convolution Layer Block

    Args:
        in_channels (int): Number of input channels.
        count (int): Number of (convolution => [BN] => ReLU) blocks to apply.
        Share (bool): Whether to share weights between the blocks.

    Structure:
        (convolution => [BN] => ReLU) * count
    """
    def __init__(self, in_channels, output_channels, count=2, Share= False):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, output_channels, kernel_size=3, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(output_channels))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(count):
            layers.append(nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(output_channels))
            layers.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*layers)

        if Share:
            for i in range(3, len(self.conv), 3):  # 3의 배수 인덱스를 반복
                self.conv[i].weight = nn.Parameter(self.conv[0].weight.clone())
                self.conv[i].weight.requires_grad = False


    def forward(self, x):
        return self.conv(x)
