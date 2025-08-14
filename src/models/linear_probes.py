import torch
import torch.nn as nn
from typing import Tuple

class LinearProbe(nn.Module):
    """
    A set of simple linear classifiers (probes) for different tasks
    (age, gender, race) operating on a shared input feature vector.

    Args:
        input_dim (int): Dimensionality of the input feature vector.
        age_classes (int): Number of classes for the age prediction task.
        gender_classes (int): Number of classes for the gender prediction task.
        race_classes (int): Number of classes for the race prediction task.
    """
    def __init__(self, input_dim: int, age_classes: int, gender_classes: int, race_classes: int):
        super().__init__()

        # Define separate linear layers for each classification task
        self.age_classifier = nn.Linear(input_dim, age_classes)
        self.gender_classifier = nn.Linear(input_dim, gender_classes)
        self.race_classifier = nn.Linear(input_dim, race_classes)

        # Store config parameters if needed for reference, but not strictly necessary
        self.input_dim = input_dim
        self.age_classes = age_classes
        self.gender_classes = gender_classes
        self.race_classes = race_classes


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the classifiers.

        Args:
            x (torch.Tensor): Input feature tensor, shape (batch_size, input_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the raw
                logit outputs for each task: (age_logits, gender_logits, race_logits).
                The order matches the classifier definition order.
        """
        age_logits = self.age_classifier(x)
        gender_logits = self.gender_classifier(x)
        race_logits = self.race_classifier(x)

        return age_logits, gender_logits, race_logits