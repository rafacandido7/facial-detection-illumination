# facial-detection-illumination

Comparação de três detectores faciais (Haar Cascade, HOG+SVM, YuNet) sob variações de iluminação, com e sem pré-processamento CLAHE.

**Autores:** Rafael Cândido · Lucas Tourinho · Gabriella Pereira · Lucas Rocha  
**Disciplina:** Processamento de Imagens — IDP

---

## Sobre

Detectores faciais performam bem em condições ideais, mas degradam sob baixa luminosidade, iluminação lateral e superexposição. Este projeto investiga:

- Como Haar Cascade, HOG+SVM e YuNet se comportam em 4 condições de iluminação distintas
- Se o pré-processamento CLAHE melhora (ou não) cada método
- Precisão, recall, F1-score e tempo de inferência por detector

| Detector     | Tipo                | Biblioteca  |
| ------------ | ------------------- | ----------- |
| Haar Cascade | Clássico            | OpenCV      |
| HOG + SVM    | ML clássico         | dlib        |
| YuNet        | CNN (deep learning) | OpenCV DNN  |

---

## Estrutura

```
.
├── dataset/
│   ├── proprio/          # Fotos capturadas pelos autores (4 condições de iluminação)
│   │   ├── boa_iluminacao/
│   │   ├── baixa_luz/
│   │   ├── lateral/
│   │   └── superexposicao/
│   └── lfw/              # Subset público LFW (Labeled Faces in the Wild)
├── src/
│   └── main.py           # MVP: roda os 3 detectores com/sem CLAHE numa imagem
├── models/
│   └── face_detection_yunet_2023mar.onnx   # baixado automaticamente no 1º run
├── article/
│   └── main.tex          # Artigo no template IEEEtran
├── requirements.txt
└── README.md
```

---

## Como rodar

**1. Clone o repositório**

```bash
git clone https://github.com/rafacandido7/facial-detection-illumination.git
cd facial-detection-illumination
```

**2. Crie o ambiente virtual e instale as dependências**

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> `dlib` pode exigir `cmake` instalado no sistema. No macOS: `brew install cmake`.

**3. Execute o MVP**

```bash
python src/main.py
```

Abre janela com comparação lado a lado (sem CLAHE | com CLAHE), bounding boxes coloridas:
- **Azul** → Haar Cascade
- **Verde** → HOG + SVM
- **Vermelho** → YuNet

O modelo YuNet (~2 MB) é baixado automaticamente no primeiro uso.

**Imagem customizada:**

```bash
python src/main.py caminho/para/imagem.jpg
```

---

## Dataset LFW (opcional)

```bash
# Kaggle CLI
kaggle datasets download jessicali9530/lfw-dataset
unzip lfw-dataset.zip -d dataset/lfw/
```

O repositório já inclui os arquivos de metadados CSV. As imagens não são versionadas por tamanho.

---

## Dependências

```
opencv-python>=4.8.0
scikit-image>=0.21.0
dlib>=19.24.0
numpy>=1.24.0
matplotlib>=3.7.0
pandas>=2.0.0
Pillow>=10.0.0
tqdm>=4.65.0
```

---

## Artigo

Artigo científico em `article/main.tex`, template IEEEtran.  
Título: *Análise Comparativa de Técnicas de Detecção Facial sob Variações de Iluminação para Aplicações em Segurança*

---

## Referências

- Viola & Jones — *Rapid object detection using a boosted cascade of simple features* (CVPR, 2001)
- Dalal & Triggs — *Histograms of oriented gradients for human detection* (CVPR, 2005)
- Wu et al. — *YuNet: A tiny millisecond-level face detector* (MIR, 2023)
- Zuiderveld — *Contrast limited adaptive histogram equalization* (Graphics Gems IV, 1994)
- Huang et al. — *Labeled Faces in the Wild* (UMass Tech Report, 2007)
