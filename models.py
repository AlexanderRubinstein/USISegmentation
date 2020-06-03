import torch


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

class out_conv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(out_conv, self).__init__()
        
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.conv(x)

class Fourier2d(torch.nn.Module):
    def __init__(self, image_size):
        super(Fourier2d, self).__init__()

        self.w = torch.ones(image_size, requires_grad=True)

    def forward(self, x):
        w = self.w.unsqueeze(-1).repeat(x.shape[0], 1, 1, 1, 2).to(x.device)
        zero_complex_part = torch.zeros_like(x)
        
        ft_x = torch.fft(torch.cat([x.unsqueeze(-1), zero_complex_part.unsqueeze(-1)], dim=-1), signal_ndim=3, normalized=True)
        ift = torch.ifft(ft_x * w, signal_ndim=3, normalized=True)

        return torch.sqrt(torch.pow(ift[..., 0], 2) + torch.pow(ift[..., 1], 2))

class NLFourier2d(torch.nn.Module):
    def __init__(self, image_size):
        super(NLFourier2d, self).__init__()

        self.w = torch.zeros(image_size, requires_grad=True)

    def forward(self, x):
        w = self.w.repeat(x.shape[0], 1, 1, 1).to(x.device)
        zero_complex_part = torch.zeros_like(x)
        
        ft_x = torch.fft(torch.cat([x.unsqueeze(-1), zero_complex_part.unsqueeze(-1)], dim=-1), signal_ndim=3, normalized=True)
        w = torch.pow(torch.sqrt(torch.pow(ft_x[..., 0], 2) + torch.pow(ft_x[..., 1], 2)), w)
        ift = torch.ifft(ft_x * w.unsqueeze(-1), signal_ndim=3, normalized=True)

        return torch.sqrt(torch.pow(ift[..., 0], 2) + torch.pow(ift[..., 1], 2))

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
#         x = Fourier2d(x.shape[1:])(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        
        bottom = self.bottom_bridge(down2)
        
        up1 = self.up1(bottom, down2)
        up2 = self.up2(up1, down1)
        
        return self.outconv(up2)


class attention(torch.nn.Module):
    def __init__(self, shape):
        super(attention, self).__init__()
        self.W = torch.randn(1, *shape)

    def forward(self, x):

        W = (self.W.expand(x.shape[0], -1, -1, -1)).to(x.device)
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

