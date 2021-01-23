import torch
from torch import nn
import torch.nn.functional as F

class SiameseNet(nn.Module):
   def __init__(self, input_size, output_size, hidden_size, num_layers):
      super().__init__()
      
      self.num_layers = num_layers
      self.hidden_size = hidden_size

      self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False, dropout=0.5)

      self.linear1 = nn.Sequential(
         #nn.Dropout(0.5),
         nn.Linear(hidden_size, hidden_size),
         nn.ReLU()
      )

      self.linear2 = nn.Sequential(
         #nn.Dropout(0.5),
         nn.Linear(hidden_size, output_size)
      )
      
   def old_forward(self, data):
      res = []
      
      x = data
      _, x = self.rnn(x)
      # x = torch.cat((x[0], x[1]), 1)
      x = x[0]

      return x

   def forward(self, data):
      res = []
      for i in range(2): # Siamese nets; sharing weights
         x = data[:,i,:,:]
         _, x = self.rnn(x)
         # x = torch.cat((x[0], x[1]), 1)
         x = x[0]

         # feed output to a fully connected layer
         
         x = self.linear1(x)
         res.append(x)
         
      res = torch.abs(res[1] - res[0])
      res = self.linear2(res)
      return res