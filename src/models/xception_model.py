import torch
import torch.nn as nn
import timm
import logging

class XceptionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.num_classes = config["model"]["num_classes"]
        self.dropout_rate = config["model"]["dropout_rate"]
        self.config = config  # Store config for later use
        
        # Build appearance feature classifier network
        # Takes pre-extracted features and processes them through MLP layers
        self.appearance_classifier = nn.Sequential(
            nn.Linear(config["model"]["feature_dim"], config["model"]["classifier_hidden_size"]),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(config["model"]["classifier_hidden_size"], config["model"]["classifier_hidden_size"])
        )
        
        # Create flow feature backbone using ResNet18 architecture
        # Modified to accept 2-channel optical flow input
        self.flow_backbone = timm.create_model(
            config["model"]["flow_architecture"], 
            pretrained=True, 
            num_classes=0,  # Remove classification layer
            in_chans=2      # 2 channels for optical flow (x,y directions)
        )
        
        # Flow feature classifier network
        self.flow_classifier = nn.Sequential(
            nn.Linear(config["model"]["flow_feature_dim"], config["model"]["classifier_hidden_size"]),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )
        
        # Final classifier that combines both feature types
        self.final_classifier = nn.Linear(2 * config["model"]["classifier_hidden_size"], self.num_classes)
        
        self.device = torch.device(config["training"]["device"])

    def forward(self, data):
        # Process appearance features from pre-extracted Xception features
        app_features = data["appearance"]  # Shape: (batch, num_frames, feature_dim)
        self.logger.debug(f"Appearance shape: {app_features.shape}, mean: {app_features.mean().item():.4f}, std: {app_features.std().item():.4f}")
        
        # Apply temporal pooling by averaging across frames
        app_features = torch.mean(app_features, dim=1)  # Shape: (batch, feature_dim)
        
        # Pass through appearance classifier network
        app_out = self.appearance_classifier(app_features)  # Shape: (batch, classifier_hidden_size)
        
        # Check if optical flow features are available and enabled in config
        if "flow" in data and self.config.get("model", {}).get("use_flow", True):
            # Process optical flow features
            flow_features = data["flow"]  # Shape: (batch, num_frames-1, image_size, image_size, 2)
            self.logger.debug(f"Flow shape: {flow_features.shape}, mean: {flow_features.mean().item():.4f}, std: {flow_features.std().item():.4f}")
            
            # Reshape flow features for CNN processing
            batch, num_flows, h, w, c = flow_features.shape
            flow_features = flow_features.view(batch * num_flows, c, h, w)  # Reshape for CNN input
            
            # Extract flow features through backbone
            flow_out = self.flow_backbone(flow_features)  # Shape: (batch * num_flows, flow_feature_dim)
            
            # Reshape back to batch dimension and apply temporal pooling
            flow_out = flow_out.view(batch, num_flows, -1).mean(dim=1)  # Average over time
            
            # Pass through flow classifier network
            flow_out = self.flow_classifier(flow_out)  # Shape: (batch, classifier_hidden_size)
            
            # Combine appearance and flow features for final classification
            combined = torch.cat([app_out, flow_out], dim=1)  # Shape: (batch, 2 * classifier_hidden_size)
            outputs = self.final_classifier(combined)
        else:
            # Fallback to appearance features only when flow is unavailable
            self.logger.warning("Flow features not available, using appearance features only")
            # Create a zero tensor for flow features to maintain correct dimensions
            flow_placeholder = torch.zeros_like(app_out)
            combined = torch.cat([app_out, flow_placeholder], dim=1)  # Shape: (batch, 2 * classifier_hidden_size)
            outputs = self.final_classifier(combined)
        
        return outputs