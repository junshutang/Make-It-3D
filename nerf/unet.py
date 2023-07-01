import torch
import torch.nn as nn
import torch.nn.functional as F

_assert_if_size_mismatch = True

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, normalization=nn.BatchNorm2d):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
                                    normalization(out_channels),
                                    nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size, padding=1),
                                    normalization(out_channels),
                                    nn.ReLU())

    def forward(self, inputs, **kwargs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class GatedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, padding_mode='reflect', act_fun=nn.ELU, normalization=nn.BatchNorm2d):
        super().__init__()
        self.pad_mode = padding_mode
        self.filter_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        n_pad_pxl = int(self.dilation * (self.filter_size - 1) / 2)

        # this is for backward campatibility with older model checkpoints
        self.block = nn.ModuleDict(
            {
                'conv_f': nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=n_pad_pxl),
                'act_f': act_fun(),
                'conv_m': nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=n_pad_pxl),
                'act_m': nn.Sigmoid(),
                'norm': normalization(out_channels)
            }
        )

    def forward(self, x, *args, **kwargs):
        features = self.block.act_f(self.block.conv_f(x))
        mask = self.block.act_m(self.block.conv_m(x))
        output = features * mask
        output = self.block.norm(output)

        return output


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_block=BasicBlock):
        super().__init__()

        self.conv = conv_block(in_channels, out_channels)
        self.down = nn.AvgPool2d(2, 2)

    def forward(self, inputs, mask=None):
        outputs = self.down(inputs)
        outputs = self.conv(outputs, mask=mask)
        return outputs


class UpsampleBlock(nn.Module):
    def __init__(self, out_channels, upsample_mode, same_num_filt=False, conv_block=BasicBlock):
        super().__init__()
        self.out_channels = out_channels

        num_filt = out_channels if same_num_filt else out_channels * 2
        if upsample_mode == 'deconv':
            self.up = nn.ConvTranspose2d(num_filt, out_channels, 4, stride=2, padding=1)
            self.conv = conv_block(out_channels * 2, out_channels, normalization=Identity)
        elif upsample_mode=='bilinear' or upsample_mode=='nearest':
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode=upsample_mode),
                                    # Before refactoring, it was a nn.Sequential with only one module.
                                    # Need this for backward compatibility with model checkpoints.
                                    nn.Sequential(
                                        nn.Conv2d(num_filt, out_channels, 3, padding=1)
                                        )
                                    )
            self.conv = conv_block(out_channels * 2, out_channels, normalization=Identity)
        else:
            assert False

    def forward(self, inputs1, inputs2):
        in1_up = self.up(inputs1)
        if (inputs2.size(2) != in1_up.size(2)) or (inputs2.size(3) != in1_up.size(3)):
            if _assert_if_size_mismatch:
                raise ValueError(f'input2 size ({inputs2.shape[2:]}) does not match upscaled inputs1 size ({in1_up.shape[2:]}')
            diff2 = (inputs2.size(2) - in1_up.size(2)) // 2
            diff3 = (inputs2.size(3) - in1_up.size(3)) // 2
            inputs2_ = inputs2[:, :, diff2 : diff2 + in1_up.size(2), diff3 : diff3 + in1_up.size(3)]
        else:
            inputs2_ = inputs2
        output= self.conv(torch.cat([in1_up, inputs2_], 1))
        return output


class UNet(nn.Module):
    r""" Rendering network with UNet architecture and multi-scale input."""
    def __init__(
        self,
        num_input_channels=3, 
        num_output_channels=3,
        feature_scale=4,
        upsample_mode='bilinear',
        last_act='sigmoid',
        conv_block='gated'
    ):
        super().__init__()

        self.feature_scale = feature_scale

        self.num_input_channels = num_input_channels

        if conv_block == 'basic':
            self.conv_block = BasicBlock
        elif conv_block == 'gated':
            self.conv_block = GatedBlock
        else:
            raise ValueError('bad conv block {}'.format(conv_block))

        filters = [64, 128, 256, 512, 1024]
        filters = [x // self.feature_scale for x in filters]

        # norm_layer = get_norm_layer(norm_layer)

        self.start = self.conv_block(self.num_input_channels, filters[0])
        
        self.down1 = DownsampleBlock(filters[0], filters[1] - self.num_input_channels, conv_block=self.conv_block)
        self.down2 = DownsampleBlock(filters[1], filters[2] - self.num_input_channels, conv_block=self.conv_block)

        self.up2 = UpsampleBlock(filters[1], upsample_mode, conv_block=self.conv_block)
        self.up1 = UpsampleBlock(filters[0], upsample_mode, conv_block=self.conv_block)

        # Before refactoring, it was a nn.Sequential with only one module.
        # Need this for backward compatibility with model checkpoints.
        self.final = nn.Sequential(
            nn.Conv2d(filters[0], num_output_channels, 1)
        )
        if last_act == 'sigmoid':
            self.final = nn.Sequential(self.final, nn.Sigmoid())
        elif last_act == 'tanh':
            self.final = nn.Sequential(self.final, nn.Tanh())

    def forward(self, inputs, **kwargs):
        masks = [None] * len(inputs)        
        in64 = self.start(inputs[0], mask=masks[0])
        mask = masks[1] 
        down1 = self.down1(in64, mask)
        down1 = torch.cat([down1, inputs[1]], 1)

        mask = masks[2] 
        down2 = self.down2(down1, mask)
        down2 = torch.cat([down2, inputs[2]], 1)

        up_= down2
        up_ = self.up2(up_, down1)
        up1 = self.up1(up_, in64)
        return self.final(up1)
