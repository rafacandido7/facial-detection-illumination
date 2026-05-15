# Phase 2: Datasets & Annotations - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-13
**Phase:** 02-datasets-annotations
**Areas discussed:** LFW annotations, LFW subset selection, Own dataset, Annotation tooling & GT format

---

## LFW Annotations

| Option | Description | Selected |
|--------|-------------|----------|
| Usar LFW-a público | Coordenadas de alinhamento facial para todas as imagens. Converter para bbox sem anotação manual. | ✓ |
| Anotar manualmente no subconjunto | labelme em ~50-100 imagens. 2-4h de trabalho humano. | |
| FDDB annotations | ~2845 imagens com sobreposição parcial com LFW. | |

**User's choice:** LFW-a público
**Notes:** Economia crítica de tempo dado o prazo de 1 dia.

---

## LFW Subset Selection

| Option | Description | Selected |
|--------|-------------|----------|
| ~100 imagens | Suficiente para métricas, rápido de rodar. | ✓ |
| ~50 imagens | Mínimo para artigo acadêmico. | |
| ~200 imagens | Mais robusto estatisticamente, 2x mais lento. | |

**User's choice:** ~100 imagens

| Option | Description | Selected |
|--------|-------------|----------|
| Subpastas por condição (parent.name) | Criar bright/, dark/, lateral/, overexposed/ e organizar imagens. | ✓ |
| Uma condição única 'lfw' | LFW só para métricas gerais; dataset próprio para análise por condição. | |
| IQAL automático | Brightness analysis automático para classificar. | |

**User's choice:** Subpastas por condição — mesmo esquema do dataset próprio.

---

## Own Dataset

| Option | Description | Selected |
|--------|-------------|----------|
| Capturamos antes de rodar | Equipe captura ~20 imgs antes da Phase 2. | ✓ |
| Só LFW se der aperto | Dataset próprio opcional. | |

**User's choice:** Capturamos antes de rodar.

| Option | Description | Selected |
|--------|-------------|----------|
| dataset/proprio/bright/, dark/, lateral/, overexposed/ | Inglês, consistente com LFW subset. | ✓ |
| dataset/proprio/boa_iluminacao/, baixa_luminosidade/, lateral/, superexposicao/ | Português. | |

**User's choice:** Inglês — consistente com subconjunto LFW.

---

## Annotation Tooling & GT Format

| Option | Description | Selected |
|--------|-------------|----------|
| labelme + script de conversão | labelme gera JSON por imagem; script converte para gt.json unificado. | ✓ |
| Anotar direto em JSON à mão | Manual, rápido para poucas imagens, propenso a erros. | |

**User's choice:** labelme + script.

| Option | Description | Selected |
|--------|-------------|----------|
| Um arquivo GT por dataset | dataset/proprio/gt.json e dataset/lfw_subset/gt.json | ✓ |
| Um arquivo GT unificado | gt_all.json para ambos | |

**User's choice:** Um por dataset — mais simples como argumento CLI.

| Option | Description | Selected |
|--------|-------------|----------|
| Script olhos→bbox com padding fixo | LFW-a dá coords dos olhos; face_width = dist(eyes)*2.5 | ✓ |
| FDDB como GT para overlap LFW | Ellipse annotations com mapeamento de nomes | |

**User's choice:** Script Python com fórmula de olhos.

---

## Claude's Discretion

- Lógica exata de conversão LFW-a (formato dos landmark files, padding)
- Critério de seleção das 100 imagens LFW
- Script de conversão labelme → gt.json

## Deferred Ideas

None.
