# Roadmap: Análise Comparativa de Detecção Facial sob Variações de Iluminação

## Overview

Partindo de um pipeline base já funcional (3 detectores + CLAHE em `src/main.py`), o projeto completa a infraestrutura de avaliação com métricas reais, anota e executa dois datasets, gera visualizações para o artigo e escreve as seções faltantes — tudo dentro de 2 dias, culminando em um PDF IEEEtran entregável.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Evaluation Pipeline** - Transformar `src/main.py` em pipeline de avaliação batch com IoU, Precision/Recall/F1, timing e exportação CSV
- [ ] **Phase 2: Datasets & Annotations** - Capturar dataset próprio, anotar ground truth em ambos os datasets e rodar pipeline completo em cada um
- [ ] **Phase 3: Results & Visualizations** - Gerar gráficos e imagens anotadas a partir dos dados reais do experimento
- [ ] **Phase 4: Article Completion** - Escrever seções faltantes do artigo e compilar PDF final sem erros

## Phase Details

### Phase 1: Evaluation Pipeline
**Goal**: O sistema processa batches de imagens nos 3 detectores × 2 passes e produz métricas (IoU, P/R/F1, tempo) exportáveis em CSV
**Depends on**: Nothing (MVP `src/main.py` já existe)
**Requirements**: PIPE-01, PIPE-02, PIPE-03, PIPE-04, PIPE-05
**Success Criteria** (what must be TRUE):
  1. Rodar o pipeline em um diretório de imagens produz resultados para todos os 3 detectores × raw e CLAHE sem intervenção manual por imagem
  2. Para cada detecção, um valor de IoU é calculado contra o ground truth correspondente e um limiar de 0.5 classifica como TP/FP/FN
  3. O sistema imprime ou exporta Precision, Recall e F1-score por detector e por condição de iluminação
  4. O tempo médio de inferência por detector é registrado e agregado ao final da execução
  5. Um arquivo CSV (ou JSON) com todos os resultados é salvo em disco e pode ser aberto em planilha
**Plans**: 3 plans

Plans:
- [x] 01-01-PLAN.md — Create src/detectors.py: detector pre-init pattern, fix dlib 140ms timing bug, shared CLAHE function
- [x] 01-02-PLAN.md — Create src/metrics.py: IoU, greedy TP/FP/FN matching, micro-averaged P/R/F1 aggregation
- [x] 01-03-PLAN.md — Create src/evaluate.py: batch CLI entrypoint, tqdm loop, GT JSON loading, CSV export

### Phase 2: Datasets & Annotations
**Goal**: Ambos os datasets estão anotados com ground truth e têm resultados gerados pelo pipeline
**Depends on**: Phase 1
**Requirements**: OWN-01, OWN-02, OWN-03, LFW-01, LFW-02, LFW-03
**Success Criteria** (what must be TRUE):
  1. Existe um conjunto de ~20 imagens próprias organizadas em 4 subdiretórios de condição de iluminação
  2. Cada imagem própria tem bounding box de ground truth anotada (arquivo JSON/XML compatível com o pipeline)
  3. Um subconjunto do LFW representativo de variações de iluminação está selecionado e com ground truth mapeado
  4. O pipeline roda em ambos os datasets e gera arquivos CSV com resultados — sem erros de execução
**Plans**: 3 plans

Plans:
- [ ] 02-01-PLAN.md — Build dataset/lfw_subset/: select 25 images/condition via lfw_attributes.txt, Haar auto-annotate, write gt.json
- [ ] 02-02-PLAN.md — Create dataset/proprio/ structure, write convert_labelme.py, human capture checkpoint, produce gt.json
- [ ] 02-03-PLAN.md — Validate both gt.json files, run evaluate.py on lfw_subset and proprio, verify CSVs

### Phase 3: Results & Visualizations
**Goal**: Os resultados numéricos viram figuras prontas para inserir no artigo
**Depends on**: Phase 2
**Requirements**: VIZ-01, VIZ-02, VIZ-03
**Success Criteria** (what must be TRUE):
  1. Existe um gráfico de barras (PNG/PDF) comparando Precision, Recall e F1 por detector × condição de iluminação, gerado a partir dos CSVs reais
  2. Existem imagens salvas em `results/` com as bounding boxes de detecção e ground truth sobrepostas, distinguíveis visualmente (cores diferentes)
  3. Existe um gráfico comparando o tempo médio de inferência entre os 3 detectores, gerado a partir dos dados de timing reais
**Plans**: TBD

### Phase 4: Article Completion
**Goal**: O artigo IEEEtran está completo com todas as seções escritas, figuras inseridas e compila em PDF sem erros
**Depends on**: Phase 3
**Requirements**: ART-01, ART-02, ART-03, ART-04, ART-05, ART-06
**Success Criteria** (what must be TRUE):
  1. A Seção II (Fundamentação Teórica) cobre Haar Cascade, HOG+SVM via dlib (corrigido de scikit-image), YuNet e CLAHE com referências
  2. A Seção IV (Resultados) contém as tabelas e gráficos gerados na Phase 3, com dados reais do experimento
  3. As Seções V (Discussão) e VI (Conclusão e Agradecimentos) estão escritas e interpretam os resultados
  4. `make` no diretório `article/` compila o PDF sem erros ou warnings fatais, e o PDF resultante está legível e completo
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Evaluation Pipeline | 0/3 | Not started | - |
| 2. Datasets & Annotations | 0/3 | Not started | - |
| 3. Results & Visualizations | 0/? | Not started | - |
| 4. Article Completion | 0/? | Not started | - |
