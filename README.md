# facial-detection-illumination

Comparação de três detectores faciais (Haar Cascade, HOG+SVM, YuNet) sob quatro condições de iluminação, com e sem pré-processamento CLAHE. Inclui pipeline de avaliação completo, dois datasets anotados e artigo científico no template IEEEtran.

**Autores:** Rafael Cândido · Lucas Tourinho · Gabriella Pereira · Lucas Rocha  
**Disciplina:** Processamento de Imagens — IDP (8º semestre, Ciência da Computação)

---

## Sobre

Detectores faciais performam bem em condições ideais, mas degradam sob baixa luminosidade, iluminação lateral e superexposição. Este projeto compara três métodos que representam paradigmas distintos da área:

| Detector     | Paradigma           | Biblioteca  | Sensibilidade a iluminação |
| ------------ | ------------------- | ----------- | -------------------------- |
| Haar Cascade | Cascata de features | OpenCV      | Alta (depende de contraste local fixo) |
| HOG + SVM    | Gradientes + SVM    | dlib        | Média (normalização em blocos ajuda) |
| YuNet        | CNN leve            | OpenCV DNN  | Menor (aprende features dos dados) |

Para cada detector, avaliamos duas passagens: imagem bruta e imagem pré-processada com CLAHE. Métricas: Precision, Recall, F1-score (IoU ≥ 0.5) e tempo médio de inferência.

---

## Resultados principais (LFW subset, n=25/condição)

| Detector | Boa luz F1 | Escuro F1 | Lateral F1 | Superexp. F1 | Tempo médio |
|----------|-----------|-----------|------------|-------------|-------------|
| Haar (raw) | 0.962 | 0.893 | **0.980** | 0.962 | ~3.4 ms |
| HOG (raw) | 0.898 | 0.720 | 0.939 | 0.923 | ~9.2 ms |
| YuNet (raw) | 0.923 | 0.717 | 0.962 | 0.906 | **~2.1 ms** |

CLAHE ajudou o HOG em escuro (+3pp) mas **piorou** o Haar em escuro (-12pp), amplificando falsos positivos.

---

## Estrutura

```
.
├── src/
│   ├── detectors.py      # Haar, HOG+SVM, YuNet + CLAHE
│   ├── metrics.py        # IoU, match de detecções, P/R/F1
│   ├── evaluate.py       # CLI: roda avaliação em batch, gera CSV
│   └── main.py           # Demo interativo: 3 detectores lado a lado
├── scripts/
│   ├── build_lfw_subset.py    # Seleciona 100 imgs do LFW por condição de iluminação
│   ├── build_proprio_gt.py    # Auto-anota GT do dataset próprio (HOG → YuNet → Haar)
│   ├── convert_labelme.py     # Converte anotações labelme → gt.json
│   └── generate_figures.py    # Gera figuras para o artigo (bar charts F1 + tempo)
├── dataset/
│   ├── lfw_subset/       # 100 imgs LFW (25/condição) + gt.json
│   │   ├── bright/
│   │   ├── dark/
│   │   ├── lateral/
│   │   └── overexposed/
│   └── proprio/          # 19 imgs capturadas pelos autores + gt.json
│       ├── bright/
│       ├── dark/
│       ├── lateral/
│       └── overexposed/
├── results/
│   ├── lfw_subset/       # raw_results.csv + summary.csv
│   ├── proprio/          # raw_results.csv + summary.csv
│   └── figures/          # fig_f1_bar.pdf, fig_time_bar.pdf
├── models/
│   └── face_detection_yunet_2023mar.onnx   # baixado automaticamente no 1º run
├── article/
│   ├── main.tex          # Artigo IEEEtran (completo, compila com make)
│   └── Makefile
├── TO-DO.md
└── requirements.txt
```

---

## Como rodar

**1. Instale as dependências**

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> `dlib` requer `cmake`. No macOS: `brew install cmake`.

**2. Demo interativo (uma imagem)**

```bash
python src/main.py                        # abre câmera ou imagem padrão
python src/main.py caminho/para/imagem.jpg
```

Mostra bounding boxes coloridas: Azul = Haar, Verde = HOG, Vermelho = YuNet.

**3. Avaliação em batch**

```bash
# LFW subset (já vem anotado)
python src/evaluate.py \
  --dataset-dir dataset/lfw_subset \
  --gt-file dataset/lfw_subset/gt.json \
  --out-dir results

# Dataset próprio
python src/evaluate.py \
  --dataset-dir dataset/proprio \
  --gt-file dataset/proprio/gt.json \
  --out-dir results
```

Output: `results/{dataset}/raw_results.csv` e `summary.csv`.

**4. Gerar figuras**

```bash
python scripts/generate_figures.py
# → results/figures/fig_f1_bar.pdf
# → results/figures/fig_time_bar.pdf
```

**5. Compilar o artigo**

```bash
cd article && make
# → article/main.pdf
```

---

## Datasets

### LFW subset
Selecionados automaticamente pelo `build_lfw_subset.py` a partir do [LFW-deepfunneled](https://vis-www.cs.umass.edu/lfw/), usando atributos de iluminação do `lfw_attributes.txt`. GT gerado com Haar Cascade (mesma estratégia do alinhamento do LFW).

```bash
# Para baixar o LFW completo:
kaggle datasets download jessicali9530/lfw-dataset
unzip lfw-dataset.zip -d dataset/lfw/
python scripts/build_lfw_subset.py
```

### Dataset próprio
19 fotos capturadas pelos autores com smartphone (1920×1080), 4 pessoas, 4 condições de iluminação. GT auto-anotado com HOG como detector primário.

```bash
# Para adicionar novas imagens ao próprio:
python scripts/build_proprio_gt.py   # re-gera gt.json
```

---

## Observações de implementação

- **YuNet em alta resolução**: o `score_threshold` padrão do OpenCV (0.9) rejeita faces válidas em imagens ≥1080p. Ajustamos para 0.7 em `src/detectors.py`.
- **GT e viés**: o GT do LFW usa Haar; o GT do próprio usa HOG. Ambos introduzem viés em favor do detector anotador — limitação conhecida, discutida no artigo.
- **Dataset próprio pequeno**: 4–6 imagens por condição é insuficiente para conclusões estatísticas. Os resultados do próprio servem como validação qualitativa.

---

## Artigo

`article/main.tex` — template IEEEtran, compilável com `make`.

**Título:** *Análise Comparativa de Técnicas de Detecção Facial sob Variações de Iluminação para Aplicações em Segurança*

---

## Referências

- Viola & Jones — *Rapid object detection using a boosted cascade of simple features* (CVPR, 2001)
- Dalal & Triggs — *Histograms of oriented gradients for human detection* (CVPR, 2005)
- Wu et al. — *YuNet: A tiny millisecond-level face detector* (MIR, 2023)
- Zuiderveld — *Contrast limited adaptive histogram equalization* (Graphics Gems IV, 1994)
- Huang et al. — *Labeled Faces in the Wild* (UMass Tech Report, 2007)
