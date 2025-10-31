# 🐕 SmartPetID Demo

Uma implementação de demonstração do sistema SmartPetID para identificação biométrica de animais de estimação usando Deep Learning.

## 📋 Descrição

O SmartPetID é um sistema de identificação biométrica que utiliza características faciais únicas de cães para criar "impressões digitais" digitais. Este projeto demonstra como o sistema pode ser usado para:

- **Identificar cães perdidos**: Comparar fotos de cães encontrados com uma base de dados de animais registrados
- **Verificar identidade**: Confirmar se um cão é realmente quem afirma ser
- **Reunir famílias**: Ajudar a reunir animais perdidos com seus donos

## 🏗️ Arquitetura

O sistema utiliza:

- **Backbone**: Swin Transformer (swin_base_patch4_window7_224) para extração de características
- **Global Descriptor**: Rede neural para agregação de características espaciais
- **Embedding**: Vetores de 1024 dimensões com normalização L2
- **Similaridade**: Cálculo de similaridade coseno para comparação

## 🚀 Funcionalidades

- ✅ **Extração de características biométricas** de imagens de cães
- ✅ **Geração de embeddings únicos** de 1024 dimensões
- ✅ **Comparação de similaridade** entre diferentes imagens
- ✅ **Carregamento de modelo pré-treinado** com checkpoint otimizado
- ✅ **Preprocessamento automático** de imagens
- ✅ **Interface simples** para demonstração

## 📦 Instalação

### Pré-requisitos

- Python 3.9+
- PyTorch
- torchvision
- timm
- PIL (Pillow)
- NumPy

### Configuração do Ambiente

1. **Clone o repositório**:
```bash
git clone <repository-url>
cd smartpetid_demo
```

2. **Crie um ambiente virtual**:
```bash
python -m venv .venv
source .venv/bin/activate  # No Windows: .venv\Scripts\activate
```

3. **Instale as dependências**:
```bash
pip install torch torchvision torchaudio
pip install timm pillow numpy
```

4. **Baixe o modelo pré-treinado**:
   - Coloque o arquivo `swin224_stage2.pth` no diretório raiz do projeto

## 🖼️ Preparação das Imagens

Coloque suas imagens de teste no diretório do projeto:

- `dog1_a.png` - Imagem do cão registrado
- `dog2.png` - Imagem do cão a ser identificado

**Requisitos das imagens**:
- Formato: PNG, JPG, JPEG
- Resolução: Qualquer (será redimensionada para 224x224)
- Conteúdo: Foco no rosto/focinho do cão
- Qualidade: Boa iluminação e nitidez

## 🎯 Uso

### Execução Básica

```bash
python smartpetid_demo.py
```

### Exemplo de Saída

```
=== SmartPetID Demo - Implementação 100% Fiel ===

1. Criando modelo...

2. Carregando checkpoint...
Carregando checkpoint: swin224_stage2.pth
✅ Checkpoint carregado!

3. Configurando preprocessamento...

4. Testando com imagens...
   Gerando embedding para cão registrado...
   ✅ Embedding gerado: (1024,) (dim=1024)
   Gerando embedding para cão encontrado...
   ✅ Embedding gerado: (1024,) (dim=1024)

5. Calculando similaridade...
   Similaridade coseno: 0.956789
   ✅ MATCH! (similaridade > 0.8)
   🐕 Cão encontrado corresponde ao registrado!

=== Demonstração concluída ===
```

### Personalização

Para usar suas próprias imagens, modifique as linhas no arquivo `smartpetid_demo.py`:

```python
# Altere os nomes dos arquivos
registered_embedding = generate_biometric_signature(
    model, "sua_imagem_registrada.jpg", preprocess
)

found_embedding = generate_biometric_signature(
    model, "sua_imagem_encontrada.jpg", preprocess
)
```

## 📊 Interpretação dos Resultados

### Similaridade Coseno

- **0.9 - 1.0**: Match muito provável (mesmo cão)
- **0.8 - 0.9**: Match provável (verificação recomendada)
- **0.6 - 0.8**: Similaridade moderada (possível match)
- **< 0.6**: Provavelmente cães diferentes

### Threshold Padrão

O sistema usa um threshold de **0.8** por padrão. Valores acima indicam um match positivo.

## 🔧 Configurações Técnicas

### Modelo

- **Arquitetura**: Swin Transformer Base
- **Input**: Imagens 224x224 RGB
- **Output**: Embeddings 1024D normalizados L2
- **Configuração GD**: SM (Small-Medium)

### Preprocessamento

- **Redimensionamento**: 224x224 pixels
- **Normalização**: ImageNet statistics
  - Mean: [0.485, 0.456, 0.406]
  - Std: [0.229, 0.224, 0.225]

## 📁 Estrutura do Projeto

```
smartpetid_demo/
├── .gitignore              # Arquivos ignorados pelo Git
├── README.md               # Este arquivo
├── smartpetid_demo.py      # Script principal
├── swin224_stage2.pth      # Modelo pré-treinado
├── dog1_a.png             # Imagem de exemplo (cão registrado)
├── dog1_b.png             # Imagem de exemplo (mesmo cão)
├── dog2.png               # Imagem de exemplo (cão diferente)
└── .venv/                 # Ambiente virtual (ignorado)
```

## 🧪 Testes

### Teste com Mesmo Cão

```bash
# Usando dog1_a.png e dog1_b.png (mesmo cão)
# Resultado esperado: Similaridade > 0.9
```

### Teste com Cães Diferentes

```bash
# Usando dog1_a.png e dog2.png (cães diferentes)
# Resultado esperado: Similaridade < 0.8
```

## ⚠️ Limitações

- **Qualidade da imagem**: Imagens borradas ou mal iluminadas podem afetar a precisão
- **Ângulo**: Fotos frontais do rosto/focinho funcionam melhor
- **Raça**: O modelo pode ter performance variável entre diferentes raças
- **Idade**: Mudanças significativas na aparência podem afetar a identificação

## 🔬 Detalhes Técnicos

### Arquitetura do Modelo

1. **Patch Embedding**: Converte imagem em patches 4x4
2. **Swin Layers**: 4 estágios de atenção hierárquica
3. **Layer Normalization**: Normalização final
4. **Global Descriptor**: Agregação espacial com configuração SM
5. **L2 Normalization**: Normalização final do embedding

### Pipeline de Inferência

```python
# Sequência de processamento
x = backbone.patch_embed(x)    # [B, H, W, C]
x = backbone.layers(x)         # [B, 7, 7, 1024]
x = backbone.norm(x)           # [B, 7, 7, 1024]
x = x.permute(0, 3, 1, 2)      # [B, 1024, 7, 7]
x = main_modules(x)            # [B, 1024] L2-normalized
```

## 🤝 Contribuição

Para contribuir com o projeto:

1. Fork o repositório
2. Crie uma branch para sua feature (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto é fornecido para fins de demonstração e pesquisa.