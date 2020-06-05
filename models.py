import torch
from crfseg import CRF

from convolution_lstm import ConvLSTM

class double_conv(torch.nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(double_conv, self).__init__()
        
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, mid_channels, kernel_size, stride, padding),
            torch.nn.BatchNorm2d(mid_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(mid_channels, out_channels, kernel_size, stride, padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class down_step(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(down_step, self).__init__()
        
        self.pool = torch.nn.MaxPool2d(kernel_size=2)
        self.conv = double_conv(in_channels, out_channels, out_channels, kernel_size, stride, padding)
    
    def forward(self, x):
        return self.conv(self.pool(x))

class up_step(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(up_step, self).__init__()

        self.up = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = double_conv(in_channels, out_channels, out_channels, kernel_size, stride, padding)

    def forward(self, from_up_step, from_down_step):
        upsampled = self.up(from_up_step)
        x = torch.cat([from_down_step, upsampled], dim=1)
        return self.conv(x)

class up_step_triple_cat(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(up_step_triple_cat, self).__init__()

        self.up = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = double_conv(3 * out_channels, out_channels, out_channels, kernel_size, stride, padding)

    def forward(self, from_up_step, from_down_step, from_prev):
        upsampled = self.up(from_up_step)
        x = torch.cat([from_down_step, from_prev, upsampled], dim=1)
        return self.conv(x)

class out_conv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(out_conv, self).__init__()
        
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        )

    def forward(self, x):
        return self.conv(x)

class UNet(torch.nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        
        self.down1 = double_conv(n_channels, 32, 32)
        self.down2 = down_step(32, 64)

        self.bottom_bridge = down_step(64, 128)

        self.up1 = up_step(128, 64)
        self.up2 = up_step(64, 32)
        
        self.outconv = out_conv(32, n_classes)

    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        
        bottom = self.bottom_bridge(down2)
        
        up1 = self.up1(bottom, down2)
        up2 = self.up2(up1, down1)
        
        return self.outconv(up2)

class attention(torch.nn.Module):
    def __init__(self, shape):
        super(attention, self).__init__()
        self.W = torch.randn(1, *shape).to('cuda')

    def forward(self, x):

        W = self.W.expand(x.shape[0], -1, -1, -1)
        return W * x


class Unet_with_attention(torch.nn.Module):
    def __init__(self, n_channels, n_classes, height, width):
        super(Unet_with_attention, self).__init__()

        self.down1 = double_conv(n_channels, 32, 32)
        self.att1 = attention((32, height, width))

        self.down2 = down_step(32, 64)
        self.att2 = attention((64, height//2, width//2))

        self.bottom_bridge = down_step(64, 128)

        self.up1 = up_step(128, 64)
        self.up2 = up_step(64, 32)

        self.outconv = out_conv(32, n_classes)

    def forward(self, x):
        #         x = Fourier2d(x.shape[1:])(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)

        bottom = self.bottom_bridge(down2)

        up1 = self.up1(bottom, self.att2(down2))
        up2 = self.up2(up1, self.att1(down1))

        return self.outconv(up2)


class UNetTC(torch.nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNetTC, self).__init__()
        
        self.down1 = double_conv(n_channels, 32, 32)
        self.down2 = down_step(32, 64)

        self.bottom_bridge = down_step(64, 128)

        self.up1 = up_step_triple_cat(128, 64)
        self.up2 = up_step_triple_cat(64, 32)
        
        self.outconv = out_conv(32, n_classes)

    def forward(self, x):
        down1 = self.down1(x)
        down1_prev = torch.cat([down1[0][None, ...], down1[:-1]], dim=0)
        down2 = self.down2(down1)
        down2_prev = torch.cat([down2[0][None, ...], down2[:-1]], dim=0)
        
        bottom = self.bottom_bridge(down2)
        
        up1 = self.up1(bottom, down2, down2_prev)
        up2 = self.up2(up1, down1, down1_prev)
        
        return self.outconv(up2)

class UNetTC3(torch.nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNetTC3, self).__init__()
        
        self.down1 = double_conv(n_channels, 32, 32)
        self.down2 = down_step(32, 64)
        self.down3 = down_step(64, 128)

        self.bottom_bridge = down_step(128, 256)

        self.up1 = up_step_triple_cat(256, 128)
        self.up2 = up_step_triple_cat(128, 64)
        self.up3 = up_step_triple_cat(64, 32)
        
        self.outconv = out_conv(32, n_classes)

    def forward(self, x):
        down1 = self.down1(x)
        down1_prev = torch.cat([down1[0][None, ...], down1[:-1]], dim=0)
        down2 = self.down2(down1)
        down2_prev = torch.cat([down2[0][None, ...], down2[:-1]], dim=0)
        down3 = self.down3(down2)
        down3_prev = torch.cat([down3[0][None, ...], down3[:-1]], dim=0)
        
        bottom = self.bottom_bridge(down3)
        
        up1 = self.up1(bottom, down3, down3_prev)
        up2 = self.up2(up1, down2, down2_prev)
        up3 = self.up3(up2, down1, down1_prev)
        
        return self.outconv(up3)

class UNetTC4(torch.nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNetTC4, self).__init__()
        
        self.down1 = double_conv(n_channels, 32, 32)
        self.down2 = down_step(32, 64)
        self.down3 = down_step(64, 128)
        self.down4 = down_step(128, 256)

        self.bottom_bridge = down_step(256, 512)

        self.up1 = up_step_triple_cat(512, 256)
        self.up2 = up_step_triple_cat(256, 128)
        self.up3 = up_step_triple_cat(128, 64)
        self.up4 = up_step_triple_cat(64, 32)
        
        self.outconv = out_conv(32, n_classes)

    def forward(self, x):
        down1 = self.down1(x)
        down1_prev = torch.cat([down1[0][None, ...], down1[:-1]], dim=0)
        down2 = self.down2(down1)
        down2_prev = torch.cat([down2[0][None, ...], down2[:-1]], dim=0)
        down3 = self.down3(down2)
        down3_prev = torch.cat([down3[0][None, ...], down3[:-1]], dim=0)
        down4 = self.down4(down3)
        down4_prev = torch.cat([down4[0][None, ...], down4[:-1]], dim=0)
        
        bottom = self.bottom_bridge(down4)
        
        up1 = self.up1(bottom, down4, down4_prev)
        up2 = self.up2(up1, down3, down3_prev)
        up3 = self.up3(up2, down2, down2_prev)
        up4 = self.up4(up3, down1, down1_prev)
        
        return self.outconv(up4)


class Fourier2d(torch.nn.Module):
    def __init__(self, image_size):
        super(Fourier2d, self).__init__()

        self.w = torch.empty(image_size, requires_grad=True)
        torch.nn.init.xavier_uniform_(self.w)
        self.register_parameter(name='fourier_filter', param=torch.nn.Parameter(self.w))

    def forward(self, x):
        w = self.w.unsqueeze(-1).repeat(x.shape[0], 1, 1, 1, 2).to(x.device)
        zero_complex_part = torch.zeros_like(x)
        
        ft_x = torch.fft(torch.cat([x.unsqueeze(-1), zero_complex_part.unsqueeze(-1)], dim=-1), signal_ndim=3, normalized=True)
        ift = torch.ifft(ft_x * w, signal_ndim=3, normalized=True)
        if torch.isnan(ift[0, 0, 0, 0, 0]):
              print('BEFORE loss')

        return torch.sqrt(torch.pow(ift[..., 0], 2) + torch.pow(ift[..., 1], 2))

class NLFourier2d(torch.nn.Module):
    def __init__(self, image_size):
        super(NLFourier2d, self).__init__()

        self.w = torch.empty(image_size, requires_grad=True)
        torch.nn.init.xavier_uniform_(self.w)
        self.register_parameter(name='fourier_filter', param=torch.nn.Parameter(self.w))

    def forward(self, x):
        w = self.w.repeat(x.shape[0], 1, 1, 1).to(x.device)
        zero_complex_part = torch.zeros_like(x)
        
        ft_x = torch.fft(torch.cat([x.unsqueeze(-1), zero_complex_part.unsqueeze(-1)], dim=-1), signal_ndim=3, normalized=True)
        w = torch.pow(torch.sqrt(torch.pow(ft_x[..., 0], 2) + torch.pow(ft_x[..., 1], 2)), w)
        ift = torch.ifft(ft_x * w.unsqueeze(-1), signal_ndim=3, normalized=True)

        return torch.sqrt(torch.pow(ift[..., 0], 2) + torch.pow(ift[..., 1], 2))

class UNetFourier(torch.nn.Module):
    def __init__(self, n_channels, n_classes, image_size, fourier_layer=None):
        super(UNetFourier, self).__init__()
        
        self.down1 = double_conv(n_channels, 32, 32)
        self.down2 = down_step(32, 64)

        self.bottom_bridge = down_step(64, 128)

        self.up1 = up_step(128, 64)
        self.up2 = up_step(64, 32)
        
        self.outconv = out_conv(32, n_classes)
        
        self.fourier_layer = fourier_layer
        H, W = image_size
        # self.fl = Fourier2d((1, H, W))
        if self.fourier_layer == 'linear':
            self.fl1 = Fourier2d((32, H, W))
            self.fl2 = Fourier2d((64, H//2, W//2))
        elif self.fourier_layer == 'non-linear':
            self.fl1 = NLFourier2d((32, H, W))
            self.fl2 = NLFourier2d((64, H//2, W//2))

    def forward(self, x):
        # x = self.fl(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        
        bottom = self.bottom_bridge(down2)
        
        if self.fourier_layer is not None:
            down2 = self.fl2(down2)
            down1 = self.fl1(down1)
        
        up1 = self.up1(bottom, down2)
        up2 = self.up2(up1, down1)
        
        return self.outconv(up2)


class UNet_crf(torch.nn.Module):

    def __init__(self, n_channels, n_classes):
        super(UNet_crf, self).__init__()

        self.down1 = double_conv(n_channels, 32, 32)
        self.down2 = down_step(32, 64)

        self.bottom_bridge = down_step(64, 128)

        self.up1 = up_step(128, 64)
        self.up2 = up_step(64, 32)

        self.outconv = out_conv(32, n_classes)

        self.crf = CRF(n_spatial_dims=3)

    def forward(self, x):

        down1 = self.down1(x)
        down2 = self.down2(down1)

        bottom = self.bottom_bridge(down2)

        up1 = self.up1(bottom, down2)
        up2 = self.up2(up1, down1)

        almost_out = self.outconv(up2) # [bs, 2, h, w]
        almost_out = almost_out.reshape(*almost_out.shape, 1) # [bs, 2, h, w, 1]
        almost_out = almost_out.transpose(0, 4) # [1, 2, h, w, bs]

        out = self.crf(almost_out)

        out = out.transpose(0, 4) # [bs, 2, h, w, 1]
        out = out.squeeze(4) # [bs, 2, h, w]

        return out


class UNetCLSTM(torch.nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNetCLSTM, self).__init__()
        
        self.down1 = double_conv(n_channels, 32, 32)
        self.clstm1 = ConvLSTM(input_channels=32, hidden_channels=[64, 32, 32, 16, 32], kernel_size=3, step=5, effective_step=[4]).cuda()
        self.down2 = down_step(32, 64)
        self.clstm2 = ConvLSTM(input_channels=64, hidden_channels=[64, 32, 32, 32, 64], kernel_size=3, step=5, effective_step=[4]).cuda()

        self.bottom_bridge = down_step(64, 128)

        self.up1 = up_step(128, 64)
        self.up2 = up_step(64, 32)

        self.outconv = out_conv(32, n_classes)

    def forward(self, x):
        down1 = self.down1(x)
        outputs1, _ = self.clstm1(down1)

        down2 = self.down2(down1)
        outputs2, _ = self.clstm2(down2)

        bottom = self.bottom_bridge(down2)

        up1 = self.up1(bottom, outputs2[0])
        up2 = self.up2(up1, outputs1[0])

        return self.outconv(up2)

