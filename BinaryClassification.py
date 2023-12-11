import torch.nn as nn

class BinaryClassification(nn.Module):
    def __init__(self):
        super(BinaryClassification, self).__init__()
        # Number of input features is 12.
        self.layer_1 = nn.Linear(472, 640) 
        self.layer_2 = nn.Linear(640, 120)
        self.layer_3 = nn.Linear(120, 64)
        self.layer_out = nn.Linear(64, 1) 
        
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(640)
        self.batchnorm2 = nn.BatchNorm1d(120)
        self.batchnorm3 = nn.BatchNorm1d(64)
    def init_weights(self):
        nn.init.kaiming_normal_(self.layer_1.weight)
        nn.init.kaiming_normal_(self.layer_2.weight)
        nn.init.kaiming_normal_(self.layer_3.weight)
        nn.init.kaiming_normal_(self.layer_out.weight)      
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout1(x)
        x = self.relu(self.layer_3(x))
        x = self.batchnorm3(x)
        x = self.dropout2(x)
        x = self.layer_out(x)
        
        return x
