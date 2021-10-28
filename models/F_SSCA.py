import torch.nn as nn
import torch


class Attention(nn.Module):

    def __init__(self, channel=64, ratio=8):
        super(Attention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

        self.shared_layer = nn.Sequential(
            nn.Linear(in_features=channel, out_features=channel // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=channel // ratio, out_features=channel),
            nn.ReLU(inplace=True)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, F):
        b, c, _, _ = F.size()

        F_avg = self.shared_layer(self.avg_pool(F).reshape(b, c))
        F_max = self.shared_layer(self.max_pool(F).reshape(b, c))
        M = self.sigmoid(F_avg + F_max).reshape(b, c, 1, 1)

        return F * M


class d_ssca(nn.Module):

    def __init__(self):
        super(d_ssca, self).__init__()

        self.convolution_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(9, 16), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=128)
        )

        self.max_pooling_1 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 1))

        self.attention = Attention(channel=128, ratio=16)  # 32/4   64/8   128/16

        self.output = nn.Sequential(
            nn.AdaptiveMaxPool2d(output_size=(1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=256, out_features=1),  # 64, 128, 256
            nn.Sigmoid()
        )

    def _forward_impl(self, seq_shape):
        seq_shape = seq_shape.float()

        conv_1 = self.convolution_1(seq_shape)
        pool_1 = self.max_pooling_1(conv_1)

        attention_1 = self.attention_seq(pool_1)

        return self.output(attention_1)

    def forward(self, seq_shape):
        return self._forward_impl(seq_shape)