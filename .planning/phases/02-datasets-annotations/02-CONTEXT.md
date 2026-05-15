# Phase 2: Datasets & Annotations - Context

**Gathered:** 2026-05-13
**Status:** Ready for planning

<domain>
## Phase Boundary

Anotar ground truth em dois datasets (LFW subset + dataset próprio) e rodar o pipeline
completo em ambos, produzindo CSVs com métricas reais para o artigo.

Não inclui: visualizações (Phase 3), escrita do artigo (Phase 4).

</domain>

<decisions>
## Implementation Decisions

### LFW Ground Truth

- **D-01:** Usar **LFW-a** (alignment annotations públicas) como fonte de ground truth para o LFW. Não anotar manualmente.
- **D-02:** Conversão: landmarks de olhos do LFW-a → bounding box via script Python. Fórmula: `face_width = dist(left_eye, right_eye) * 2.5`, bbox centrado nos olhos com padding proporcional.
- **D-03:** Subset de **~100 imagens** do LFW.
- **D-04:** Imagens LFW organizadas em subpastas por condição de iluminação (usando `img_path.parent.name` como `condition`). Subpastas: `bright/`, `dark/`, `lateral/`, `overexposed/` — mesmos nomes do dataset próprio.
- **D-05:** GT JSON salvo em `dataset/lfw_subset/gt.json` — um arquivo por dataset.

### Dataset Próprio

- **D-06:** Estrutura de pastas: `dataset/proprio/bright/`, `dark/`, `lateral/`, `overexposed/`. ~5 imagens por condição, ~20 total.
- **D-07:** Captura feita pela equipe antes de rodar o pipeline. Prerequisito humano.
- **D-08:** Anotação com **labelme** + script de conversão para o formato GT JSON unificado.
- **D-09:** GT JSON salvo em `dataset/proprio/gt.json` — um arquivo por dataset.

### GT JSON Format

- **D-10:** Formato esperado por `evaluate.py --gt-file` (definido em Plan 01-03):
  ```json
  {
    "dataset": "proprio",
    "iou_threshold": 0.5,
    "images": [
      {
        "file": "dataset/proprio/bright/img001.jpg",
        "condition": "bright",
        "faces": [{"x1": 120, "y1": 80, "x2": 280, "y2": 300}]
      }
    ]
  }
  ```
- **D-11:** Paths no campo `"file"` são relativos à raiz do projeto (PROJECT_ROOT), compatível com a correção CR-01 aplicada em Phase 1.

### Annotation Tooling

- **D-12:** LFW-a: script Python automatizado (sem ferramenta de UI).
- **D-13:** Dataset próprio: **labelme** para anotação manual + script Python para converter `labelme JSON → gt.json` unificado.

### Claude's Discretion

- Lógica exata de conversão LFW-a (formato dos arquivos de landmark, padding exato) — o planner/executor pesquisa o formato do LFW-a e implementa.
- Critério de seleção das 100 imagens LFW (aleatório estratificado por condição? random seed fixo?) — executor decide.
- Formato exato de saída do labelme e script de conversão — executor implementa.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase 1 Artifacts (pipeline já construído)
- `src/evaluate.py` — CLI entrypoint; aceita `--dataset-dir`, `--gt-file`, `--out-dir`, `--iou-thresh`
- `src/detectors.py` — detectores pré-inicializados
- `src/metrics.py` — IoU, match_detections, aggregate_metrics
- `.planning/phases/01-evaluation-pipeline/01-03-PLAN.md` — define GT JSON format (interfaces section)
- `.planning/phases/01-evaluation-pipeline/01-03-SUMMARY.md` — resume o que foi construído

### Dataset Structure
- `dataset/lfw/lfw-deepfunneled/lfw-deepfunneled/` — imagens LFW (5749 pessoas)
- `dataset/lfw/people.csv` — lista de pessoas e contagem de imagens

### Requirements
- `.planning/REQUIREMENTS.md` — OWN-01, OWN-02, OWN-03, LFW-01, LFW-02, LFW-03

No external specs for LFW-a format — executor must look up the LFW-a file format during implementation.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/evaluate.py` — pipeline pronto; Phase 2 apenas gera os datasets e GTs, depois chama o CLI
- `src/metrics.py:match_detections` — aceita gt_boxes list de tuplas (x1,y1,x2,y2)

### Established Patterns
- `condition` = `img_path.parent.name` — hardcoded em evaluate.py; subpastas DEVEM ter os nomes de condição corretos
- GT JSON keys são paths relativos ao PROJECT_ROOT (correção CR-01 em Phase 1)
- `results/<dataset_name>/raw_results.csv` é gerado automaticamente com `--out-dir results`

### Integration Points
- Phase 2 entrega: `dataset/lfw_subset/gt.json` + `dataset/proprio/gt.json`
- Phase 3 consome: `results/lfw_subset/raw_results.csv` + `results/proprio/raw_results.csv`

</code_context>

<specifics>
## Specific Ideas

- Nomes de condição em inglês (bright, dark, lateral, overexposed) nos dois datasets — consistência no CSV e no artigo
- ~100 imagens LFW é suficiente para métricas significativas no prazo
- LFW-a como GT evita horas de anotação manual — decisão crítica para o prazo

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 02-datasets-annotations*
*Context gathered: 2026-05-13*
