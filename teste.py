#!/usr/bin/env python3

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
import numpy as np
import os
import glob


class L2Norm(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=1)


class GlobalDescriptor(nn.Module):

    def __init__(self, p: float, eps: float = 1e-6):
        super().__init__()
        self.p = p
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 1:
            # average pooling
            return F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        if self.p == float('inf'):
            # max pooling
            return F.adaptive_max_pool2d(x, (1, 1)).flatten(1)
        # generalized mean pooling
        x = torch.clamp(x, min=self.eps)
        x = x.pow(self.p)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.pow(1.0 / self.p)
        return x.flatten(1)


class PetBiometricModel(nn.Module):

    def __init__(
        self,
        backbone_name: str = 'swin_base_patch4_window7_224_in22k',
        feature_dim: int = 512,
        gd_config: str = 'SM',
        pretrained_backbone: bool = True
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.gd_config = gd_config
        self.feature_dim = feature_dim

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained_backbone,
            num_classes=0,
            global_pool=''
        )

        # Determina quantas saídas globais teremos, baseado em gd_config
        self.num_global_descriptors = len(gd_config)
        self.backbone_out_channels = self.backbone.num_features
        if feature_dim % self.num_global_descriptors != 0:
            raise ValueError(
                f'feature_dim ({feature_dim}) deve ser múltiplo de '
                f'num_global_descriptors ({self.num_global_descriptors})'
            )

        self.global_dim = feature_dim // self.num_global_descriptors

        # Cria os global descriptors com base em gd_config
        gd_layers = []
        for c in gd_config:
            if c == 'S':
                # mean pooling
                gd_layers.append(GlobalDescriptor(p=1.0))
            elif c == 'M':
                # max pooling
                gd_layers.append(GlobalDescriptor(p=float('inf')))
            elif c == 'G':
                # generalized mean pooling com p=3 (por exemplo)
                gd_layers.append(GlobalDescriptor(p=3.0))
            else:
                raise ValueError(f'GD config "{c}" não suportado.')
        self.global_descriptors = nn.ModuleList(gd_layers)

        # Camadas lineares que transformam cada descritor global em global_dim
        self.main_modules = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.backbone_out_channels, self.global_dim, bias=False),
                nn.BatchNorm1d(self.global_dim)
            ) for _ in range(self.num_global_descriptors)
        ])

        self.l2norm = L2Norm()

    def _forward_backbone_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encapsula o forward do Swin até gerar os tokens (sem pooling global),
        lidando com diferenças entre versões dos timm.
        """
        _swin = self.backbone

        # 1) Patch embedding
        x = _swin.patch_embed(x)

        # 2) Posição absoluta (algumas versões não têm esse atributo)
        if hasattr(_swin, "absolute_pos_embed") and _swin.absolute_pos_embed is not None:
            x = x + _swin.absolute_pos_embed

        # 3) Dropout de posição (se existir)
        if hasattr(_swin, "pos_drop"):
            x = _swin.pos_drop(x)

        # 4) Blocos (layers ou stages dependendo da versão)
        if hasattr(_swin, "layers"):
            layers = _swin.layers
        elif hasattr(_swin, "stages"):
            layers = _swin.stages
        else:
            raise RuntimeError("Backbone Swin não possui 'layers' nem 'stages'.")

        for layer in layers:
            x = layer(x)

        # 5) Normalização final (se existir)
        if hasattr(_swin, "norm") and _swin.norm is not None:
            x = _swin.norm(x)

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

        feat = torch.cat(outs, dim=1)
        feat = self.l2norm(feat)
        return feat

def load_checkpoint_correctly(model: PetBiometricModel, checkpoint_path: str):
    print(f"Carregando checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    backbone_state_dict = {}
    main_modules_state_dict = {}
    global_descriptors_state_dict = {}

    print("Mapeando chaves do checkpoint...")
    for key, value in state_dict.items():
        # remove 'module.' se foi salvo com DataParallel / DDP
        if key.startswith('module.'):
            key = key[7:]

        # ignora estatísticas e coisas auxiliares
        if any(s in key for s in [
            'num_batches_tracked',
            'total_params',
            'relative_position_index',
            'attn_mask'
        ]):
            continue

        # o treino original usava features.*; aqui fazemos o remapeamento
        if key.startswith('features.1.'):
            # normalmente pos_drop, sem pesos
            continue

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

        # mapeia cabeças (main_modules) e descritores globais
        if key.startswith('main_modules.'):
            new_key = key.replace('main_modules.', '')
            main_modules_state_dict[new_key] = value
            continue

        if key.startswith('global_descriptors.'):
            new_key = key.replace('global_descriptors.', '')
            global_descriptors_state_dict[new_key] = value
            continue

    print(f"Carregando backbone: {len(backbone_state_dict)} chaves")

    # ✅ FILTRA apenas os pesos com shape compatível pra evitar size mismatch
    compatible_backbone_dict = {}
    current_backbone_state = model.backbone.state_dict()
    for k, v in backbone_state_dict.items():
        current = current_backbone_state.get(k, None)
        if current is not None and current.shape == v.shape:
            compatible_backbone_dict[k] = v

    incompatible = model.backbone.load_state_dict(
        compatible_backbone_dict,
        strict=False
    )
    missing_backbone = incompatible.missing_keys
    unexpected_backbone = incompatible.unexpected_keys

    print(f"Carregando main_modules: {len(main_modules_state_dict)} chaves")
    incompatible_main = model.main_modules.load_state_dict(
        main_modules_state_dict, strict=False
    )
    missing_main = incompatible_main.missing_keys
    unexpected_main = incompatible_main.unexpected_keys

    print(f"Carregando global_descriptors: {len(global_descriptors_state_dict)} chaves")
    incompatible_gd = model.global_descriptors.load_state_dict(
        global_descriptors_state_dict, strict=False
    )
    missing_gd = incompatible_gd.missing_keys
    unexpected_gd = incompatible_gd.unexpected_keys

    total_missing = len(missing_backbone) + len(missing_main) + len(missing_gd)
    total_unexpected = len(unexpected_backbone) + len(unexpected_main) + len(unexpected_gd)

    print("✅ Checkpoint carregado (ignorando pesos incompatíveis de shape).")
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

    # garante que está em modo eval
    model.eval()

    with torch.no_grad():
        embedding = model(input_tensor)

    emb = embedding.squeeze().numpy()
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb



def calculate_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    embedding1 = embedding1 / np.linalg.norm(embedding1)
    embedding2 = embedding2 / np.linalg.norm(embedding2)
    return float(np.dot(embedding1, embedding2))


def extract_dog_id_from_filename(path: str) -> str:
    """
    Extrai o ID do cachorro a partir do nome do arquivo.
    Exemplo:
        dog1_1.jpeg -> "1"
        dog23_2.png -> "23"
    """
    fname = os.path.basename(path)
    name, _ = os.path.splitext(fname)
    parts = name.split("_")
    dog_part = parts[0].lower()  # "dog1"
    if dog_part.startswith("dog"):
        return dog_part[3:]
    return dog_part


def list_images_in_dir(directory: str):
    """
    Lista imagens em um diretório para extensões comuns.
    """
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(files)


def build_registry(model: PetBiometricModel, preprocess: transforms.Compose, cadastro_dir: str):
    """
    Usa TODAS as imagens de cada cachorro em train/cadastro para construir
    uma assinatura biométrica média (embedding médio normalizado) por cão.
    """
    image_paths = list_images_in_dir(cadastro_dir)
    if not image_paths:
        raise RuntimeError(f"Nenhuma imagem encontrada em {cadastro_dir}")

    per_dog_embs = {}
    per_dog_paths = {}

    for img_path in image_paths:
        dog_id = extract_dog_id_from_filename(img_path)
        emb = generate_biometric_signature(model, img_path, preprocess)

        per_dog_embs.setdefault(dog_id, []).append(emb)
        per_dog_paths.setdefault(dog_id, []).append(img_path)

    registry_embs = {}
    for dog_id, emb_list in per_dog_embs.items():
        embs = np.stack(emb_list, axis=0)
        mean_emb = embs.mean(axis=0)
        norm = np.linalg.norm(mean_emb)
        if norm > 0:
            mean_emb = mean_emb / norm
        registry_embs[dog_id] = mean_emb

    return registry_embs, per_dog_paths


def verify_directory(model: PetBiometricModel,
                     preprocess: transforms.Compose,
                     registry_embs,
                     verificacao_dir: str,
                     threshold: float = None,
                     top_k: int = 3):
    """
    Para cada imagem em train/verificacao:
      - calcula embedding
      - compara com todos os cães cadastrados
      - NÃO assume que o nome do arquivo contém o ID verdadeiro
      - imprime a correlação (similaridade) com o melhor cão
        e o top-K de correlações.

    Se threshold não for None:
      - se max_sim < threshold -> "desconhecido"
    """
    image_paths = list_images_in_dir(verificacao_dir)
    if not image_paths:
        raise RuntimeError(f"Nenhuma imagem encontrada em {verificacao_dir}")

    dog_ids = list(registry_embs.keys())
    registry_matrix = np.stack([registry_embs[d] for d in dog_ids], axis=0)

    print("\n--- Verificação 1:N (pasta 'verificacao' com nomes aleatórios) ---")
    for img_path in image_paths:
        emb = generate_biometric_signature(model, img_path, preprocess)

        # Similaridade com cada cão cadastrado
        sims = []
        for emb_reg in registry_matrix:
            sims.append(calculate_similarity(emb, emb_reg))
        sims = np.array(sims)

        # Melhor match
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        best_dog_id = dog_ids[best_idx]

        # Threshold opcional
        predicted_label = best_dog_id
        if threshold is not None and best_sim < threshold:
            predicted_label = "desconhecido"

        print(f"\nImagem: {os.path.basename(img_path)}")
        if predicted_label == "desconhecido":
            print(f"  Melhor correlação abaixo do threshold ({threshold:.3f}). "
                  f"Score máximo = {best_sim:.4f} → DESCONHECIDO")
        else:
            print(f"  Melhor match: dog{best_dog_id}  | correlação = {best_sim:.4f}")

        # Top-K correlações
        sorted_idx = np.argsort(-sims)  # ordem decrescente
        k = min(top_k, len(sorted_idx))
        print("  Top correlações:")
        for rank in range(k):
            idx = sorted_idx[rank]
            dog = dog_ids[idx]
            score = float(sims[idx])
            print(f"    Top{rank+1}: dog{dog}  | correlação = {score:.4f}")



def main():
    """
    Script principal:
      - carrega o modelo e o checkpoint
      - usa TODAS as imagens em train/cadastro para "treinar" (criar template) de cada cão
      - verifica todas as imagens em train/verificacao contra o banco de cadastros
    """
    print("=== SmartPetID – Cadastro + Verificação 1:N ===")

    base_dir = "train"
    cadastro_dir = os.path.join(base_dir, "cadastro")
    verificacao_dir = os.path.join(base_dir, "verificacao")
    checkpoint_path = "swin224_stage2.pth"
    threshold = 0.75  # ajuste se quiser ser mais/menos rígido

    print(f"1. Criando modelo (backbone Swin + Dual Global Descriptor)...")
    model = PetBiometricModel(
        backbone_name="swin_base_patch4_window7_224_in22k",
        feature_dim=512,
        gd_config="SM",
        pretrained_backbone=True
    )

    print(f"2. Carregando checkpoint de '{checkpoint_path}'...")
    model = load_checkpoint_correctly(model, checkpoint_path)
    model.eval()
    print("3. Construindo pipeline de pré-processamento...")
    preprocess = build_preprocess()

    print(f"4. Construindo banco de cadastros a partir de: {cadastro_dir}")
    registry_embs, registry_imgs = build_registry(model, preprocess, cadastro_dir)
    print(f"   ✅ Total de cães cadastrados: {len(registry_embs)}")

    print(f"5. Rodando verificação 1:N nas imagens de: {verificacao_dir}")
    threshold = 0.75
    top_k = 3
    metrics = verify_directory(
        model=model,
        preprocess=preprocess,
        registry_embs=registry_embs,
        verificacao_dir=verificacao_dir,
        threshold=threshold,
        top_k=top_k
    )

    print("\n=== Processo concluído. Correlações calculadas para todas as imagens de verificação. ===")


if __name__ == '__main__':
    main()
