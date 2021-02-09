import torch
from torch import nn
import torch.nn.functional as F

class SiameseNet(nn.Module):
   def __init__(self, input_size, hidden_size, num_layers):
      super().__init__()
      
      self.num_layers = num_layers
      self.hidden_size = hidden_size

      self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False, dropout=0.5)
      self.bn1 = nn.BatchNorm1d(input_size)

   def forward(self, data):
      device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
      x = data.to(device)
      
      x = x.permute(0, 2, 1)
      x = self.bn1(x)
      x = x.permute(0, 2, 1)

      output, (x, c_n) = self.rnn(x)
      
      return x[1,:,:]