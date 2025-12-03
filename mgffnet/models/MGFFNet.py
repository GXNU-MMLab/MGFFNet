import os
import timm
import torch
from timm.models.vision_transformer import Block
from torchvision import models
from torch import nn
from einops import rearrange
from tool.mfe import MFE
from tool.maf import MAF
from tool.haf import HAF
from tool.ife import IFE
class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []
class MGFFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = timm.create_model('vit_base_patch8_224', pretrained=True)

        self.save_output = SaveOutput()
        hook_handles = []
        for layer in self.vit.modules():
            if isinstance(layer, Block):
                handle = layer.register_forward_hook(self.save_output)
                hook_handles.append(handle)

        from resnet_modify import resnet50 as resnet_modifyresnet
        modelpretrain = models.resnet50(pretrained=True)
        torch.save(modelpretrain.state_dict(), 'modelpretrain')
        self.model = resnet_modifyresnet()
        self.model.load_state_dict(torch.load('modelpretrain'), strict=True)
        os.remove("modelpretrain")

        self.avg8 = nn.AvgPool2d((8, 8))
        self.avg4 = nn.AvgPool2d((4, 4))
        self.avg2 = nn.AvgPool2d((2, 2))

        self.conv2 = nn.Conv2d(512, 256, 1, 1, 0)
        self.conv3 = nn.Conv2d(1024, 256, 1, 1, 0)
        self.conv4 = nn.Conv2d(2048, 256, 1, 1, 0)
        self.convt = nn.Conv2d(768, 1024, 1, 1, 0)

        self.enhance1 = MFE(256, [7, 5, 3])
        self.enhance2 = MFE(512, [7, 5, 3])
        self.enhance3 = MFE(1024, [3, 2, 1])
        self.enhance4 = MFE(2048, [3, 2, 1])
        self.ife = IFE(256, 8)
        self.fc_score = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1),
            nn.ReLU()
        )
        self.fc_weight = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

        self.maf=MAF(1024,8)
        self.haf=HAF(2048,2048)
        self.conv_=nn.Conv2d(2048,1024,3,1,1)

    def forward(self, x):
        _x = self.vit(x)
        self.save_output.outputs.clear()
        _x = rearrange(_x, 'b (h w) c -> b c h w', h=28, w=28)
        _x = self.convt(self.avg4(_x))
        out, layer1, layer2, layer3, layer4 = self.model(x)

        layer2_t = self.conv2(self.enhance2(layer2))
        layer3_t = self.conv3(self.enhance3(layer3))
        layer4_t = self.conv4(self.enhance4(layer4))

        layer1_ = self.avg8(self.enhance1(layer1))
        layer2_ = self.avg4(layer2_t)
        layer3_ = self.avg2(layer3_t)
        l3 = self.ife(layer4_t, layer3_)
        l2 = self.ife(l3, layer2_)
        l1 = self.ife(l2, layer1_)
        layers = torch.cat((l1, l2, l3, layer4_t), dim=1)
        layers_out = self.maf(layers, _x)
        layers_out2=self.conv_(self.haf(torch.cat((layers, _x),dim=1)))
        layers_out=layers_out2+layers_out

        layers_out = rearrange(layers_out, 'b c h w -> b (h w) c', h=7, w=7)
        score = torch.tensor([]).cuda()
        for i in range(layers_out.shape[0]):  # (8,49,3840)
            f = self.fc_score(layers_out[i])
            w = self.fc_weight(layers_out[i])
            _s = (torch.sum(f * w) / torch.sum(w)).to('cuda:0')
            score = torch.cat((score, _s.unsqueeze(0)), 0).to('cuda:0')
        return score
