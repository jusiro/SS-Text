import torch
import numpy as np

# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Adapter(torch.nn.Module):
    def __init__(self, initial_prototypes, logit_scale, adapter="ZS"):
        super().__init__()

        # Init
        self.adapt_strategy = adapter
        self.logit_scale = torch.tensor(logit_scale)
        self.logit_scale.requires_grad = False
        self.text_embeddings_avg = initial_prototypes

        # Set strategy for classifier head initialization
        self.init = "random" if ("RI" in adapter) else "zero_shot"

        # Set classifier
        if "LP++" in adapter:
            self.adapter = LPpp(self.text_embeddings_avg)
        elif "TIPAd" in adapter:
            self.adapter = TIPAd(self.text_embeddings_avg)
        elif "TaskRes" in adapter:
            self.adapter = TaskResHead(self.text_embeddings_avg, self.logit_scale)
        elif "ClipAdapt" in adapter:
            self.adapter = CLIPAdaptHead(self.text_embeddings_avg, self.logit_scale)
        else:
            self.adapter = LinearProbeHead(self.text_embeddings_avg, self.logit_scale, init=self.init)

        # move to device
        self.to(device).float()

    def forward(self, x):

        # Forward classifier
        out = self.adapter(x)

        return out

    def reset(self):

        # Set classifier
        self.adapter = LinearProbeHead(self.text_embeddings_avg, self.logit_scale, init=self.init)

        # move to device
        self.to(device).float()


class LinearProbeHead(torch.nn.Module):
    def __init__(self, zero_shot_prot, logit_scale, init="zero_shot"):
        super().__init__()
        self.logit_scale = logit_scale.data.clone()
        self.logit_scale.requires_grad = False
        self.init = init
        self.zero_shot_prot = zero_shot_prot.clone()

        if init == "zero_shot":
            self.prototypes = zero_shot_prot.clone()
        else:
            self.prototypes = torch.nn.init.kaiming_normal_(torch.empty(zero_shot_prot.shape))

        # Trainable parameters
        self.prototypes = torch.nn.Parameter(self.prototypes)

        # Keep temperature scaling as in pre-training
        self.logit_scale = logit_scale.data.clone()
        self.logit_scale.requires_grad = False

    def forward(self, features):

        # Get trained prototype
        prototypes = self.prototypes.to(device)

        # Unit hypersphere normalization
        image_features_norm = features / features.norm(dim=-1, keepdim=True)
        prototypes_norm = prototypes / prototypes.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = image_features_norm @ prototypes_norm.t() * logit_scale

        return logits


class TaskResHead(torch.nn.Module):
    def __init__(self, zero_shot_prot, logit_scale):
        super().__init__()
        self.logit_scale = logit_scale.data.clone()
        self.logit_scale.requires_grad = False
        self.zero_shot_prot = zero_shot_prot.clone()

        # Residual prototype
        self.prototypes = torch.nn.Parameter(torch.zeros_like(self.zero_shot_prot.clone()))

        # Keep temperature scaling as in pre-training
        self.logit_scale = logit_scale.data.clone()
        self.logit_scale.requires_grad = False

    def forward(self, features):

        # Get trained prototype
        prototypes = self.zero_shot_prot.clone().to(device) + self.prototypes.to(device)

        # Unit hypersphere normalization
        image_features_norm = features / features.norm(dim=-1, keepdim=True)
        prototypes_norm = prototypes / prototypes.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = image_features_norm @ prototypes_norm.t() * logit_scale

        return logits


class CLIPAdaptHead(torch.nn.Module):
    def __init__(self, zero_shot_prot, logit_scale):
        super().__init__()
        self.logit_scale = logit_scale.data.clone()
        self.logit_scale.requires_grad = False
        self.zero_shot_prot = zero_shot_prot.clone().to(device)
        self.features_length = zero_shot_prot.shape[-1]

        # Residual prototype
        self.adapter = torch.nn.Sequential(torch.nn.Linear(self.features_length, self.features_length//4, bias=False),
                                           torch.nn.Linear(self.features_length // 4, self.features_length, bias=False))
        self.alpha = 0.2

        # Keep temperature scaling as in pre-training
        self.logit_scale = logit_scale.data.clone()
        self.logit_scale.requires_grad = False

    def forward(self, features):

        # Residual adapter features
        features = (1-self.alpha) * features + self.alpha * self.adapter(features)

        # Normalize
        image_features_norm = features / features.norm(dim=-1, keepdim=True)
        prototypes_norm = self.zero_shot_prot / self.zero_shot_prot.norm(dim=-1, keepdim=True)

        # Logits
        logit_scale = self.logit_scale.exp()
        logits = image_features_norm @ prototypes_norm.t() * logit_scale

        return logits


class LPpp(torch.nn.Module):
    def __init__(self, zero_shot_prot):
        super().__init__()

        # Textual prototypes
        self.zero_shot_prot = zero_shot_prot.clone().to(device)
        self.zero_shot_prot /= self.zero_shot_prot.clone().norm(dim=-1, keepdim=True)

        # Visual classifier
        self.classifier = torch.nn.Linear(int(zero_shot_prot.shape[1]), int(zero_shot_prot.shape[0]), bias=True)

        # Blending parameter between image and text knowledge
        self.alpha_vec = torch.autograd.Variable(0 * torch.ones(1, int(zero_shot_prot.shape[1])))
        self.alpha_vec.requires_grad = True

    def forward(self, features):

        # Unit hypersphere normalization
        image_features_norm = features / features.norm(dim=-1, keepdim=True)
        prototypes_norm = self.zero_shot_prot.clone() / self.zero_shot_prot.clone().norm(dim=-1, keepdim=True)

        # Vision logits
        vision_logits = self.classifier(image_features_norm)

        # Textual logits
        text_logits = image_features_norm @ prototypes_norm.t()

        # Combination
        logits = vision_logits + torch.ones(features.shape[0], 1).to(features.dtype).cuda() @ self.alpha_vec * text_logits

        return logits


class TIPAd(torch.nn.Module):
    def __init__(self, zero_shot_prot):
        super().__init__()

        # Textual prototypes
        self.zero_shot_prot = zero_shot_prot.clone().to(device)
        self.zero_shot_prot /= self.zero_shot_prot.clone().norm(dim=-1, keepdim=True)

        # Visual classifier
        self.alpha = 1.0
        self.beta = 1.0

    def forward(self, features):

        # Unit hypersphere normalization
        image_features_norm = features / features.norm(dim=-1, keepdim=True)
        prototypes_norm = self.zero_shot_prot.clone() / self.zero_shot_prot.clone().norm(dim=-1, keepdim=True)

        # Vision logits
        affinity = image_features_norm @ self.cache_keys
        cache_logits = torch.exp(((-1) * (self.beta - self.beta * affinity))) @ self.cache_values

        # Textual logits
        text_logits = image_features_norm @ prototypes_norm.t()

        # Combination
        logits = text_logits + cache_logits * self.alpha

        return logits