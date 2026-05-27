"""Multi-task outcome prediction for TCM EMR."""
import torch.nn as nn

class OutcomePredictor(nn.Module):
    """复诊结局预测器 — 多任务(再入院/疗效/证候转归)"""
    def __init__(self, input_dim=128, n_outcomes=3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_outcomes)
        )
    
    def forward(self, x):
        return self.fc(x)
