import torch.nn as nn
class KeyLengthCNN(nn.Module):
    def __init__(self, input_size, output_size=23, dropout_prob=0.3):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.01),
            nn.BatchNorm1d(64),
        )
        
        self.fc_block = nn.Sequential(
            nn.Linear(64 * input_size, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_prob),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_prob),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_block(x)
        x = x.flatten(start_dim=1)
        return self.fc_block(x)