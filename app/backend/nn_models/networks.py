# app/nn_models/networks.py
#
# RESPONSABILIDAD: Definir las arquitecturas de las redes neuronales.
# Estas clases deben ser IDÉNTICAS a las del notebook de entrenamiento.
# Si cambia una sola capa, el load_state_dict() fallará o dará predicciones
# incorrectas silenciosamente.
#
# ARQUITECTURAS:
#   AirbnbMLP    → Red densa para el Modo 1 (solo datos tabulares)
#                  Entrada: vector procesado por ColumnTransformer
#                  Salida:  1 neurona (precio €/noche)
#
#   MultimodalMLP → Red de fusión para el Modo 2 (tabular + visual)
#                   Entrada: concatenación de [30 features tabulares | 512 embedding ResNet34]
#                   Salida:  1 neurona (precio €/noche)

import torch
import torch.nn as nn

class AirbnbMLP(nn.Module):
    """
    Red neuronal densa para predicción tabular (Modo 1).
    Arquitectura: 128 → 64 → 32 → 1
    BatchNorm + ReLU + Dropout(0.2) en cada capa oculta para regularización.
    """
    def __init__(self, input_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(32, 1)  # Salida lineal: precio en €
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MultimodalMLP(nn.Module):
    """
    Red de fusión multimodal (Modo 2).
    
    Recibe dos tensores separados (tabular y visual) y los concatena
    antes de pasarlos por la red de fusión:
    
    
    Arquitectura de fusión: 542 → 256 → 128 → 32 → 1
    """
    def __init__(self, fusion_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 32),
            nn.ReLU(),

            nn.Linear(32, 1),  # Salida lineal: precio en €
        )

    def forward(self, tab: torch.Tensor, vis: torch.Tensor) -> torch.Tensor:
        # La fusión ocurre aquí: concatenamos ambos vectores en la dimensión 1
        x = torch.cat([tab, vis], dim=1)  # (1,30) + (1,512) → (1,542)
        return self.net(x)