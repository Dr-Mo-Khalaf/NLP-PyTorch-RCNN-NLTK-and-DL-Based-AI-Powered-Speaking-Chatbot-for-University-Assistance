import torch
import torch.nn as nn
import torch.nn.functional as F
## add prediction 


class chatModel_v3_extended(nn.Module):
    def __init__(self, input_feature, hidden_feature, out_feature):
        super(chatModel_v3_extended, self).__init__()
        
        # Layers
        self.layer1 = nn.Linear(input_feature, hidden_feature)
        self.layer2 = nn.Linear(hidden_feature, hidden_feature)
        self.layer3 = nn.Linear(hidden_feature, hidden_feature)  # New Layer
        self.layer4 = nn.Linear(hidden_feature, out_feature)
        
        # Activation function
        self.relu = nn.ReLU()
        
        # Batch Normalization and Dropout
        self.batchnorm1 = nn.BatchNorm1d(hidden_feature)
        self.batchnorm2 = nn.BatchNorm1d(hidden_feature)
        self.batchnorm3 = nn.BatchNorm1d(hidden_feature)  # New BatchNorm
        
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # First layer
        x = self.layer1(x)
        x = self.relu(x)
        x = self.batchnorm1(x)
        x = self.dropout(x)
        
        # Second layer
        x = self.layer2(x)
        x = self.relu(x)
        x = self.batchnorm2(x)
        x = self.dropout(x)

        # Third layer (new)
        x = self.layer3(x)
        x = self.relu(x)
        x = self.batchnorm3(x)
        x = self.dropout(x)

        # Output layer
        x = self.layer4(x)
        return x
    def predict(self, x):
            with torch.no_grad():  # No gradients needed for inference
                logits = self.forward(x)
                probs = F.softmax(logits, dim=1)  # Convert logits to probabilities
                predictions = torch.argmax(probs, dim=1)  # Get the class with max probability
            return predictions


class chatModel_v3(torch.nn.Module):
    def __init__(self, input_feature, hidden_feature, out_feature):
        super(chatModel_v3, self).__init__()
        
        # Layers
        self.layer1 = nn.Linear(input_feature, hidden_feature)
        self.layer2 = nn.Linear(hidden_feature, hidden_feature)
        self.layer3 = nn.Linear(hidden_feature, out_feature)
        
        # Activation function
        self.relu = nn.ReLU()
        
        # Optional: Batch Normalization and Dropout
        self.batchnorm1 = nn.BatchNorm1d(hidden_feature)
        self.batchnorm2 = nn.BatchNorm1d(hidden_feature)
        
        self.dropout = nn.Dropout(0.5)  # Dropout probability of 50%

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.batchnorm1(x)  # Apply batch normalization
        x = self.dropout(x)  # Apply dropout

        x = self.layer2(x)
        x = self.relu(x)
        # x = self.batchnorm2(x)  # Apply batch normalization
        # x = self.dropout(x)  # Apply dropout

        x = self.layer3(x)  # Output layer
        return x

    def predict(self, x):
        with torch.no_grad():  # No gradients needed for inference
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)  # Convert logits to probabilities
            predictions = torch.argmax(probs, dim=1)  # Get the class with max probability
        return predictions






class chatModel_v2(torch.nn.Module):
    def __init__(self, input_feature, hidden_feature, out_feature):
        super(chatModel_v2, self).__init__()
        
        # Layers
        self.layer1 = nn.Linear(input_feature, hidden_feature)
        self.layer2 = nn.Linear(hidden_feature, hidden_feature)
        self.layer3 = nn.Linear(hidden_feature, out_feature)
        
        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        return x

    def predict(self, x):
        # Forward pass
        with torch.no_grad():  # No gradients needed for inference
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)  # Convert logits to probabilities
            predictions = torch.argmax(probs, dim=1)  # Get the class with max probability
        return predictions




class chatModel(torch.nn.Module):
    def __init__(self, input_feature, hidden_feature, out_feature):
        super(chatModel, self).__init__()
        
        # Layers
        self.layer1 = nn.Linear(input_feature, hidden_feature)
        self.layer2 = nn.Linear(hidden_feature, hidden_feature)
        self.layer3 = nn.Linear(hidden_feature, out_feature)
        
        # Activation function
        self.relu = nn.ReLU()
        
        # Optional: Batch Normalization and Dropout for better generalization
        self.batchnorm1 = nn.BatchNorm1d(hidden_feature)
        self.batchnorm2 = nn.BatchNorm1d(hidden_feature)
        
        self.dropout = nn.Dropout(0.5)  # Dropout probability of 50%

    def forward(self, x):
        # First layer
        x = self.layer1(x)
        x = self.relu(x)
        # x = self.batchnorm1(x)  # Apply batch norm
        
        # Second layer
        x = self.layer2(x)
        x = self.relu(x)
        # x = self.batchnorm2(x)  # Apply batch norm
        
        # # Dropout layer (regularization)
        # x = self.dropout(x)
        
        # Output layer
        x = self.layer3(x)

        return x

