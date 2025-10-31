#!/usr/bin/env python3

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
import numpy as np


class L2Norm(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normaliza ao longo do último eixo (features)
        return F.normalize(x, p=2, dim=-1)


class GlobalDescriptor(nn.Module):

    def __init__(self, p: float = 1.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4, "GlobalDescriptor espera entrada [B, C, H, W]"
        if self.p == 1:
            # média global
            return x.mean(dim=[-1, -2])
        elif self.p == float('inf'):
            # máximo global
            return torch.flatten(F.adaptive_max_pool2d(x, output_size=(1, 1)), start_dim=1)
        else:
            # generalized mean pooling (GeM)
            sum_value = x.abs().pow(self.p).mean(dim=[-1, -2])
            return torch.sign(sum_value) * (sum_value.abs().pow(1.0 / self.p))


class PetBiometricModel(nn.Module):
 

    def __init__(self, backbone_name: str = 'swin_base_patch4_window7_224_in22k',
                 feature_dim: int = 512, gd_config: str = 'SM', pretrained_backbone: bool = True):
        super().__init__()

        # Instancia modelo Swin via timm sem cabeçalho de classificação.
        _swin = timm.create_model(
            backbone_name,
            pretrained=pretrained_backbone,
            num_classes=0,
            global_pool='',
        )

        # Constrói um backbone explícito contendo os módulos necessários.
        self.backbone = nn.Module()
        self.backbone.patch_embed = _swin.patch_embed
        # pos_drop pode ser None dependendo da versão
        self.backbone.pos_drop = getattr(_swin, 'pos_drop', None)
        self.backbone.layers = _swin.layers
        self.backbone.norm = _swin.norm

        # Canais de saída do Swin-B/224
        self.backbone_out_channels = 1024

        # Configuração de descritores globais
        self.gd_config = gd_config
        n = len(gd_config)
        if feature_dim % n != 0:
            raise ValueError('feature_dim deve ser divisível pelo número de descritores')
        self.k = feature_dim // n

        # Cria os descritores globais e cabeças de projeção
        gds = []
        mains = []
        for c in gd_config:
            if c == 'S':
                p = 1
            elif c == 'M':
                p = float('inf')
            else:
                p = 3
            gds.append(GlobalDescriptor(p=p))
            mains.append(nn.Sequential(
                nn.Linear(self.backbone_out_channels, self.k, bias=False),
                L2Norm()
            ))
        self.global_descriptors = nn.ModuleList(gds)
        self.main_modules = nn.ModuleList(mains)

        self.eval()

    @torch.no_grad()
    def _forward_backbone_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propaga a entrada pelo backbone Swin e retorna os tokens.

        Alguns releases de timm retornam [B, H, W, C]; outros retornam
        [B, HW, C]. Esta função não altera a forma – a normalização e
        reshaping são tratados no método `forward`.
        """
        x = self.backbone.patch_embed(x)
        if self.backbone.pos_drop is not None:
            x = self.backbone.pos_drop(x)
        # 'layers' é uma ModuleList; é mais seguro iterar explicitamente sobre cada layer
        # para garantir compatibilidade com diferentes versões do timm
        for layer in self.backbone.layers:
            x = layer(x)
        x = self.backbone.norm(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self._forward_backbone_tokens(x)
        if tokens.dim() == 4:
            # tokens: [B, H, W, C]
            b, h, w, c = tokens.shape
            if c != self.backbone_out_channels:
                raise ValueError(f'Dimensão de canais inesperada: {c}')
            shared = tokens.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        else:
            # tokens: [B, HW, C]
            b, hw, c = tokens.shape
            h = w = int(math.sqrt(hw))
            if h * w != hw:
                raise ValueError('Número de tokens não forma grade quadrada')
            if c != self.backbone_out_channels:
                raise ValueError(f'Dimensão de canais inesperada: {c}')
            shared = tokens.permute(0, 2, 1).contiguous().view(b, c, h, w)

        outs = []
        for gd, head in zip(self.global_descriptors, self.main_modules):
            v = gd(shared)
            v = head(v)
            outs.append(v)
        emb = torch.cat(outs, dim=-1)
        emb = F.normalize(emb, p=2, dim=-1)
        return emb


def load_checkpoint_correctly(model: PetBiometricModel, checkpoint_path: str):
    print(f"Carregando checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    backbone_state_dict = {}
    main_modules_state_dict = {}
    global_descriptors_state_dict = {}

    print("Mapeando chaves do checkpoint...")
    for key, value in state_dict.items():
        if key.startswith('module.'):
            key = key[7:]
        # Ignora chaves que não são parâmetros treináveis
        if any(suffix in key for suffix in ['num_batches_tracked', 'total_ops', 'total_params', 'relative_position_index', 'attn_mask']):
            continue
        # Ignora features.1.* (positional dropout) que não possuem parâmetros
        if key.startswith('features.1.'):
            continue
        # Remapeia backbone
        if key.startswith('features.0.'):
            new_key = key.replace('features.0.', 'patch_embed.')
            backbone_state_dict[new_key] = value
            continue
        if key.startswith('features.2.'):
            new_key = key.replace('features.2.', 'layers.')
            backbone_state_dict[new_key] = value
            continue
        if key.startswith('features.3.'):
            new_key = key.replace('features.3.', 'norm.')
            backbone_state_dict[new_key] = value
            continue
        # Remapeia módulos principais e descritores globais
        if key.startswith('main_modules.'):
            new_key = key.replace('main_modules.', '')
            main_modules_state_dict[new_key] = value
            continue
        if key.startswith('global_descriptors.'):
            new_key = key.replace('global_descriptors.', '')
            global_descriptors_state_dict[new_key] = value
            continue

    print(f"Carregando backbone: {len(backbone_state_dict)} chaves")
    # Apenas parâmetros com shapes compatíveis serão carregados
    compatible_backbone_dict = {}
    for k, v in backbone_state_dict.items():
        current = model.backbone.state_dict().get(k, None)
        if current is not None and current.shape == v.shape:
            compatible_backbone_dict[k] = v
    missing_backbone, unexpected_backbone = model.backbone.load_state_dict(
        compatible_backbone_dict, strict=False
    )

    print(f"Carregando main_modules: {len(main_modules_state_dict)} chaves")
    missing_main, unexpected_main = model.main_modules.load_state_dict(
        main_modules_state_dict, strict=False
    )

    print(f"Carregando global_descriptors: {len(global_descriptors_state_dict)} chaves")
    missing_gd, unexpected_gd = model.global_descriptors.load_state_dict(
        global_descriptors_state_dict, strict=False
    )

    total_missing = len(missing_backbone) + len(missing_main) + len(missing_gd)
    total_unexpected = len(unexpected_backbone) + len(unexpected_main) + len(unexpected_gd)
    print(f"✅ Checkpoint carregado!")
    print(f"   - Chaves faltando: {total_missing}")
    print(f"   - Chaves extras: {total_unexpected}")
    return model


def build_preprocess() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def generate_biometric_signature(model: PetBiometricModel, image_path: str, preprocess: transforms.Compose) -> np.ndarray:
    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model(input_tensor)
    return embedding.squeeze().numpy()


def calculate_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    embedding1 = embedding1 / np.linalg.norm(embedding1)
    embedding2 = embedding2 / np.linalg.norm(embedding2)
    return float(np.dot(embedding1, embedding2))


def main():
    print("=== SmartPetID Demo – Implementação 100% Fiel ===")
    checkpoint_path = 'swin224_stage2.pth'
    print("\n1. Criando modelo...")
    model = PetBiometricModel(
        backbone_name='swin_base_patch4_window7_224_in22k',
        feature_dim=512,
        gd_config='SM',
        pretrained_backbone=True
    )
    print("\n2. Carregando checkpoint...")
    model = load_checkpoint_correctly(model, checkpoint_path)
    print("\n3. Configurando preprocessamento...")
    preprocess = build_preprocess()
    print("\n4. Testando com imagens...")
    reg_path = 'dog1_a.png'
    found_path = 'dog2.png'
    print("   Gerando embedding para cão registrado...")
    registered_embedding = generate_biometric_signature(model, reg_path, preprocess)
    print(f"   ✅ Embedding gerado: {registered_embedding.shape} (dim={len(registered_embedding)})")
    print("   Gerando embedding para cão encontrado...")
    found_embedding = generate_biometric_signature(model, found_path, preprocess)
    print(f"   ✅ Embedding gerado: {found_embedding.shape} (dim={len(found_embedding)})")
    print("\n5. Calculando similaridade...")
    similarity = calculate_similarity(registered_embedding, found_embedding)
    print(f"   Similaridade coseno: {similarity:.6f}")
    threshold = 0.8
    if similarity > threshold:
        print(f"   ✅ MATCH! (similaridade > {threshold})")
        print("   🐕 Cão encontrado corresponde ao registrado!")
    else:
        print(f"   ❌ NO MATCH (similaridade <= {threshold})")
        print("   🔍 Cão encontrado NÃO corresponde ao registrado")
    print("\n=== Demonstração concluída ===")
    print(f"Dimensão do embedding: {len(registered_embedding)}")
    print(f"Configuração GD: {model.gd_config}")
    print("Normalização: L2 final")


if __name__ == '__main__':
    main()
