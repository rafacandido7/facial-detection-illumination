# Requirements

**Project:** Análise Comparativa de Detecção Facial sob Variações de Iluminação  
**Version:** v1  
**Deadline:** 2026-05-14

---

## v1 Requirements

### Pipeline de Avaliação

- [ ] **PIPE-01**: Sistema processa batch de imagens (não só uma) com os 3 detectores × 2 passes (raw + CLAHE)
- [ ] **PIPE-02**: Sistema calcula IoU entre cada detecção e o ground truth correspondente
- [ ] **PIPE-03**: Sistema computa Precision, Recall e F1-score por detector × condição de iluminação (IoU ≥ 0.5)
- [ ] **PIPE-04**: Sistema registra e agrega tempo médio de inferência por detector
- [ ] **PIPE-05**: Sistema exporta resultados em CSV/JSON para uso no artigo

### Dataset Próprio

- [ ] **OWN-01**: Capturar ~20 imagens em 4 condições de iluminação (~5 por condição: frontal adequada, baixa luminosidade, iluminação lateral, superexposição)
- [ ] **OWN-02**: Anotar manualmente bounding boxes de ground truth no dataset próprio
- [ ] **OWN-03**: Rodar pipeline completo no dataset próprio e gerar resultados

### Dataset LFW

- [ ] **LFW-01**: Selecionar subconjunto representativo de variações de iluminação do LFW
- [ ] **LFW-02**: Anotar bounding boxes de ground truth no subconjunto LFW selecionado
- [ ] **LFW-03**: Rodar pipeline completo no subconjunto LFW e gerar resultados

### Visualizações e Outputs

- [ ] **VIZ-01**: Gráfico de barras comparando Precision/Recall/F1 por detector × condição
- [ ] **VIZ-02**: Imagens com detecções e ground truth sobrepostos salvas em disco (resultados/)
- [ ] **VIZ-03**: Gráfico comparando tempo médio de inferência entre os 3 detectores

### Artigo

- [ ] **ART-01**: Escrever Seção II — Fundamentação Teórica (subsections: Haar Cascade, HOG+SVM, YuNet, CLAHE)
- [ ] **ART-02**: Corrigir descrição do HOG+SVM no artigo (dlib, não scikit-image)
- [ ] **ART-03**: Preencher Seção IV — Resultados com tabelas e gráficos reais
- [ ] **ART-04**: Escrever Seção V — Discussão interpretando os resultados
- [ ] **ART-05**: Escrever Seção VI — Conclusão e Agradecimentos
- [ ] **ART-06**: Compilar artigo final em PDF sem erros LaTeX

---

## v2 Requirements (deferred)

- Dataset próprio expandido (>20 imagens por condição) — prazo inviabiliza
- Validação cruzada k-fold — escopo acadêmico atual não exige
- Interface gráfica de anotação integrada — usar labelme externo

---

## Out of Scope

- Reconhecimento facial (identificação de pessoa) — só detecção é o escopo
- Detectores além de Haar, HOG+SVM, YuNet — paper define 3
- API/deploy/servidor — projeto acadêmico, CLI only
- Migração para scikit-image HOG — dlib mantido, artigo ajustado

---

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| PIPE-01 | Phase 1 — Evaluation Pipeline | Pending |
| PIPE-02 | Phase 1 — Evaluation Pipeline | Pending |
| PIPE-03 | Phase 1 — Evaluation Pipeline | Pending |
| PIPE-04 | Phase 1 — Evaluation Pipeline | Pending |
| PIPE-05 | Phase 1 — Evaluation Pipeline | Pending |
| OWN-01 | Phase 2 — Datasets & Annotations | Pending |
| OWN-02 | Phase 2 — Datasets & Annotations | Pending |
| OWN-03 | Phase 2 — Datasets & Annotations | Pending |
| LFW-01 | Phase 2 — Datasets & Annotations | Pending |
| LFW-02 | Phase 2 — Datasets & Annotations | Pending |
| LFW-03 | Phase 2 — Datasets & Annotations | Pending |
| VIZ-01 | Phase 3 — Results & Visualizations | Pending |
| VIZ-02 | Phase 3 — Results & Visualizations | Pending |
| VIZ-03 | Phase 3 — Results & Visualizations | Pending |
| ART-01 | Phase 4 — Article Completion | Pending |
| ART-02 | Phase 4 — Article Completion | Pending |
| ART-03 | Phase 4 — Article Completion | Pending |
| ART-04 | Phase 4 — Article Completion | Pending |
| ART-05 | Phase 4 — Article Completion | Pending |
| ART-06 | Phase 4 — Article Completion | Pending |
