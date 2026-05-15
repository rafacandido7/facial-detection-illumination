# Análise Comparativa de Detecção Facial sob Variações de Iluminação

## What This Is

Pesquisa acadêmica comparando três detectores faciais (Haar Cascade, HOG+SVM via dlib, YuNet CNN) avaliados com e sem pré-processamento CLAHE, em dois datasets: LFW público e dataset próprio com 4 condições de iluminação controladas. Produto final: artigo científico no formato IEEEtran e pipeline Python reproduzível.

## Core Value

Pipeline de avaliação com métricas reais (Precision, Recall, F1, tempo) sobre ground truth anotado — sem isso, o artigo não tem resultados e não pode ser entregue.

## Requirements

### Validated

- ✓ Pipeline base com 3 detectores + CLAHE — existente em `src/main.py`
- ✓ Dataset LFW disponível em `dataset/lfw/` com metadados CSV
- ✓ Estrutura do artigo IEEEtran definida em `article/main.tex`

### Active

- [ ] Carregar anotações públicas do LFW (bounding boxes ground truth)
- [x] Implementar cálculo de IoU + métricas (Precision, Recall, F1, tempo médio) — Validated in Phase 01: evaluation-pipeline
- [ ] Selecionar subconjunto LFW representativo de variações de iluminação
- [ ] Rodar pipeline completo no LFW e gerar tabelas de resultados
- [ ] Capturar dataset próprio (~20 imgs, 4 condições de iluminação)
- [ ] Anotar dataset próprio com bounding boxes (ground truth manual)
- [ ] Rodar pipeline completo no dataset próprio e gerar resultados
- [ ] Escrever Seção II — Fundamentação Teórica (Haar, HOG+SVM, YuNet, CLAHE)
- [ ] Preencher Seção IV — Resultados (tabelas + gráficos)
- [ ] Escrever Seção V — Discussão
- [ ] Escrever Seção VI — Conclusão e Agradecimentos
- [ ] Compilar artigo final em PDF

### Out of Scope

- Reconhecimento facial (identificação de pessoa) — só detecção
- Dataset maior que ~20 imagens próprias — prazo inviabiliza
- Detectors além dos 3 definidos — escopo fechado
- Deploy/API — projeto acadêmico, CLI apenas

## Context

- **Deadline**: 2026-05-14 (2 dias a partir de 2026-05-12)
- **Equipe**: Rafael Cândido, Lucas Tourinho, Gabriella Pereira, Lucas Rocha
- **LFW**: Dataset público com anotações de bounding boxes disponíveis (LFW-a, FDDB ou similar)
- **HOG+SVM**: Usando dlib (HOG+SVM pré-treinado) — artigo será ajustado para descrever corretamente como implementação dlib, não scikit-image
- **Dataset próprio**: Ainda não capturado — ~5 imgs × 4 condições (frontal adequada, baixa luminosidade, iluminação lateral, superexposição), ambiente interno, posição e distância fixas
- **Ground truth**: LFW usa anotações públicas; dataset próprio será anotado manualmente (labelme ou similar)
- **Métricas**: IoU ≥ 0.5 para TP; Precision, Recall, F1-score, tempo médio de inferência

## Constraints

- **Prazo**: 2 dias — prioridade absoluta em pipeline de métricas e resultados LFW primeiro
- **Tech stack**: Python + OpenCV + dlib — sem mudanças de stack
- **Formato artigo**: IEEEtran conference, compilado via `article/Makefile`
- **Dataset próprio**: Captura manual pelos autores — depende de disponibilidade física

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Manter dlib para HOG+SVM | Dlib é HOG+SVM pré-treinado real; migrar scikit-image sem SVM treinado inviável no prazo | — Pending |
| LFW annotations públicas | Evita trabalho de anotação manual em centenas de imagens | — Pending |
| Priorizar LFW antes do dataset próprio | LFW já disponível; dataset próprio requer captura física | — Pending |

## Evolution

Este documento evolui a cada transição de fase.

**Após cada fase:**
1. Requirements concluídos? → Mover para Validated com referência à fase
2. Novos requisitos emergiram? → Adicionar em Active
3. Decisões tomadas? → Registrar em Key Decisions

**Após entrega:**
1. Revisar o que foi validado pelos resultados reais
2. Atualizar Core Value se necessário

---
*Last updated: 2026-05-13 after Phase 01: evaluation-pipeline complete*
