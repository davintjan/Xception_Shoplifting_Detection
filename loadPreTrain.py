import torch
import timm
import torch.nn as nn

class XceptionPretrained(nn.Module):
    def __init__(self, num_classes=2):
        super(XceptionPretrained, self).__init__()

        # Load Xception model from timm
        self.model = timm.create_model('legacy_xception', pretrained=True)  # Use 'legacy_xception' instead of 'xception'

        # Freeze all layers except the last few
        for name, param in self.model.named_parameters():
            if "fc" not in name:  # Freeze everything except the final classification layer
                param.requires_grad = False

        # Modify the final fully connected layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)
