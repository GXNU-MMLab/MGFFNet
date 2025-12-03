import torch
import torch.nn as nn
import torch.nn.functional as F
class TransformerDecoderLayer(nn.Module):
    def __init__(self, in_channels, num_heads):
        super(TransformerDecoderLayer, self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads

        assert self.head_dim * num_heads == in_channels, "in_channels must be divisible by num_heads"
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.key_conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, dilation=1,
                      bias=False)
        self.key_conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2,
                                   bias=False)
        self.key_conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=3, dilation=3,
                                   bias=False)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.out_conv = nn.Conv2d(in_channels*3, in_channels, kernel_size=1, stride=1, padding=0)


    def forward(self, x,y):
        batch_size, in_channels, height, width = x.size()

        Q = self.query_conv(x).view(batch_size, self.num_heads, self.head_dim, height * width).permute(0, 1, 3,
                                                                                                       2)  # (batch_size, num_heads, height * width, head_dim)
        K1 = self.key_conv1(y).view(batch_size, self.num_heads, self.head_dim,
                                  height * width)  # (batch_size, num_heads, head_dim, height * width)
        K2 = self.key_conv2(y).view(batch_size, self.num_heads, self.head_dim,
                                    height * width)  # (batch_size, num_heads, head_dim, height * width)
        K3 = self.key_conv3(y).view(batch_size, self.num_heads, self.head_dim,
                                    height * width)  # (batch_size, num_heads, head_dim, height * width)
        V = self.value_conv(y).view(batch_size, self.num_heads, self.head_dim, height * width).permute(0, 1, 3,
                                                                                                       2)  # (batch_size, num_heads, height * width, head_dim)
        Q_norm = F.normalize(Q, p=2, dim=-1)
        K_norm1 = F.normalize(K1, p=2, dim=1)
        cos_sim1 = torch.matmul(Q_norm, K_norm1)
        attention_weights1 = F.softmax(cos_sim1, dim=-1)
        attended_values1 = torch.matmul(attention_weights1, V)  # (batch_size, num_heads, height * width, head_dim)
        attended_values1 = attended_values1.permute(0, 1, 3, 2).contiguous().view(batch_size, in_channels, height, width)

        Q_norm = F.normalize(Q, p=2, dim=-1)
        K_norm2 = F.normalize(K2, p=2, dim=1)
        cos_sim2 = torch.matmul(Q_norm, K_norm2)  # (batch_size, num_heads, height * width, height * width)
        attention_weights2 = F.softmax(cos_sim2, dim=-1)
        attended_values2 = torch.matmul(attention_weights2, V)  # (batch_size, num_heads, height * width, head_dim)
        attended_values2 = attended_values2.permute(0, 1, 3, 2).contiguous().view(batch_size, in_channels, height, width)

        Q_norm = F.normalize(Q, p=2, dim=-1)
        K_norm3 = F.normalize(K3, p=2, dim=1)
        cos_sim3 = torch.matmul(Q_norm, K_norm3)  # (batch_size, num_heads, height * width, height * width)
        attention_weights3 = F.softmax(cos_sim3, dim=-1)
        attended_values3 = torch.matmul(attention_weights3, V)  # (batch_size, num_heads, height * width, head_dim)
        attended_values3 = attended_values3.permute(0, 1, 3, 2).contiguous().view(batch_size, in_channels, height, width)
        attended_values=torch.cat((attended_values1, attended_values2, attended_values3), dim=1)
        output = self.out_conv(attended_values)

        return output

class FeedForwardNetwork(nn.Module):
    def __init__(self, in_channels, feedforward_dim=2048):
        super(FeedForwardNetwork, self).__init__()
        self.feedforward = nn.Sequential(
            nn.Conv2d(in_channels, feedforward_dim, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(feedforward_dim, in_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        return self.feedforward(x)

class MAF(nn.Module):
    def __init__(self, in_channels, num_heads, feedforward_dim=2048):
        super(MAF, self).__init__()
        self.self_attention = TransformerDecoderLayer(in_channels, 8)
        self.feedforward = FeedForwardNetwork(in_channels, feedforward_dim)
        self.norm1 = nn.LayerNorm([in_channels, 7, 7])
        self.norm2 = nn.LayerNorm([in_channels, 7, 7])
        self.dropout1=nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x,y):
        attn_output = self.self_attention(x,y)
        out = self.dropout1(attn_output)
        out = self.norm1(out)

        ff_output = self.feedforward(out)
        out = out + self.dropout2(ff_output)
        out = self.norm2(out)

        return out