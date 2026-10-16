# EXERKINEMAP

**EXERKINEMAP** (EXERcise KINEmatics Multiomics single-cell Analysis and spatial omics holistic modeling and maPping) is a computational framework for modeling how exercise-responsive molecular signals move between cells, tissues and organs.

It combines genomic and protein language models, single-cell and spatial omics, and wearable telemetry in a **stateful virtual cell world model**: an encoder, a transition operator and a decoder, with a causal-set structure that restricts every prediction to its causal past. A forward map predicts the signaling response to an intervention, and an inverse map proposes exerkine, ligand and receptor sequences that move a cell state toward a healthy reference.

<p align="center">
  <img src="docs/figures/exerkinemap_architecture.png" alt="EXERKINEMAP architecture: genomic and protein language models, fusion into a latent state, a transition operator, wearable monitoring through DBDP and WPP, and countermeasure domains" width="760">
</p>

> **Status:** research software under active development. Generated sequences and model outputs are **hypotheses to test, not results**, and nothing here is clinical or medical advice. See [Limitations and responsible use](#limitations-and-responsible-use).

## Contents
- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Project status](#project-status)
- [Quick start](#quick-start)
- [Execution workflows](#execution-workflows)
- [Omics foundation model pipeline](#omics-foundation-model-pipeline)
- [Downstream tasks](#downstream-tasks)
- [Wearables, DBDP and countermeasure design](#wearables-dbdp-and-countermeasure-design)
- [Data sources](#data-sources)
- [Tutorials and documentation](#tutorials-and-documentation)
- [Exerkine Sequence Generation Gym](#exerkine-sequence-generation-gym)
- [Tokenization ablation and one-hot control](#tokenization-ablation-and-one-hot-control)
- [GDM candidate reference and RefSeq refresh](#gdm-candidate-reference-and-refseq-refresh)
- [Limitations and responsible use](#limitations-and-responsible-use)
- [License](#license)

## What it does

By integrating genomic and protein language models with single-cell and spatial omics, EXERKINEMAP lets you input target molecular sequences to identify and map candidate exerkines, ligands and receptors. It is optimized for exercise-responsive physiological adaptation, for example evaluating exerkine signaling in Gestational Diabetes Mellitus (GDM), using multimodal datasets from [MoTrPAC (rat and human)](https://motrpac-data.org).

The model grows along two axes:

* **A virtual cell bank** across cell types, genotypes and conditions (healthy control, GDM, T2D, CVD, obesity).
* **Composition** from cells to tissues, organs and the whole person, including the fetal–maternal interface.

## Architecture

### The pipeline

* **ExerkineRNA (gLM):** tokenizes nucleotide sequences (codons, regulatory tags) into contextual genomic embeddings.
* **ExerkineProtein (pLM):** generates and refines amino-acid representations with models such as ESM-2.
* **ExerkineMap:** aligns these sequence embeddings with spatial and single-cell omics to construct predictive ligand–receptor communication networks.

### Virtual cell world model

The world model has three coupled parts, following the structure proposed for virtual cell world models in *A world model of the virtual cell* (Cell, 2026):

| Part | What it does |
|---|---|
| **Structured encoder and cellular manifold** | modality encoders (gLM, pLM, scFM, spatial) fused by message passing over a molecular network of ligands, receptors and pathways into a tissue-indexed latent state |
| **Action-conditioned transition dynamics** | advances the state under an action (exercise, gravity, nutrition) and a wearable weight; the action enters the dynamics themselves, not only an output head |
| **Structured decoder and cross-modal consistency** | decodes RNA and protein sequences, expression, space, phenotype and ligand–receptor compatibility, tied together by translation, alignment, cycle and ligand–receptor consistency losses |

<p align="center">
  <img src="docs/figures/vcwm_architecture.png" alt="Virtual cell world model: structured encoder, action-conditioned transition and structured decoder" width="700">
</p>

The full definitions, equations and algorithms are in [`docs/exerkine_math_model.md`](docs/exerkine_math_model.md).

### Omics foundation model taxonomy

EXERKINEMAP is a **sequence-centric, cell-centric and multi-omics** foundation model system. Each row follows the common omics-FM pipeline: feature representation, embedding strategy, network architecture, applications from molecular to clinical scale.

<p align="center">
  <img src="docs/figures/omics_fm_taxonomy.png" alt="EXERKINEMAP placed in the omics foundation model taxonomy" width="760">
</p>

## Project status

| Component | Status |
|---|---|
| Sequence reference building, RefSeq manifest refresh, GDM candidate catalog | available |
| Tokenization ablation and benchmark manifest builder | available |
| Pre-training, MoTrPAC fine-tuning and prompting modules (`exerkinemap/foundation_model/`) | available |
| Exerkine Sequence Generation Gym (notebook) | available |
| scFM and spatial encoder, molecular-network fusion | in development |
| Action-conditioned transition operator and decoder with consistency losses | in development |
| DBDP / WPP / SensorLM integration and countermeasure policy | design stage |
| Tissue-indexed exerkine probabilities (MoTrPAC tissues, HuBMAP organs) | design stage |

> The table reflects the current design documents. Update it as modules land.

## Quick start

Map your target sequences and generate exerkine communication networks in a few steps.

### 1. Install dependencies

```bash
conda create -n exerkinemap python=3.11
conda activate exerkinemap
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Prepare your sequences

Place your target RNA or protein sequences into `data/raw/sequences/`. Ensure your MoTrPAC reference metadata is situated in `data/raw/metadata/`.

### 3. Run the model

Execute the preprocessing and mapping workflow to align your target sequences against the multi-omics data:

```bash
# Prepare references and build sequence embeddings
python workflows/01_download_data.py
python workflows/04_build_sequence_reference.py

# Execute the mapping pipeline
python run_exerkinemap.py --input data/raw/sequences/ --output results/
```

## Execution workflows

The repository workflows cleanly separate foundation model training stages from standard spatial-omics alignment:

* `01_download_data.py`: ingests external datasets and prepares local environments.
* `02_pretrain_omics_fm.py`: executes the self-supervised pre-training pipeline on generalized large-scale datasets.
* `03_finetune_motrpac_fm.py`: executes supervised fine-tuning using validated MoTrPAC targets.
* `04_build_sequence_reference.py`: constructs the sequence reference libraries for alignment.
* `05_prompt_specialized_tasks.py`: runs human prompts and inference logic for downstream tasks such as variant prediction or biomarker identification.

## Omics foundation model pipeline

The three-stage pipeline (`exerkinemap/foundation_model/`) parses complex, multimodal biological data into cohesive physiological representations:

1. **Pre-training (`pretraining.py`):** self-supervised learning on large, unlabeled biological data. Transformers convert nucleotide (`ExerkineRNA`) and amino-acid (`ExerkineProtein`) sequences into tokens with positional encoding, establishing a baseline understanding of sequence grammar.
2. **MoTrPAC-anchored fine-tuning (`finetuning.py`):** supervised fine-tuning anchors the learned embeddings to empirical, multi-tissue expression data from the Molecular Transducers of Physical Activity Consortium (MoTrPAC). With a small curated exerkine set, prefer frozen backbones with adapters or light heads over full fine-tuning.
3. **Prompting and execution (`prompting.py`):** human prompting and Reinforcement Learning with Human Feedback (RLHF) apply the contextualized biological grammar to downstream biomedical tasks.

## Downstream tasks

The `exerkinemap/tasks/` module isolates the model's physiological outputs:

* **Exercise-responsive biomarker discovery:** extracts digital biomarkers from the latent space and cross-references them with the Exerkine Atlas.
* **DNA and protein variant prediction:** generates exerkine sequence variants and predicts ligand–receptor binding affinities for spatial communication networks.
* **Personalized medicine and virtual cell mapping:** builds patient-specific physiological profiles by tracking causal-set propagation, and generates virtual cell state maps from a sedentary baseline to a healthy, exercise-adapted state.

## Wearables, DBDP and countermeasure design

Wearable telemetry enters the world model as exogenous, causal inputs that rescale the transition dynamics.

* **Digital Biomarkers Data Pipeline (DBDP):** device harmonization (`devicely`), posture and activity recognition (DeepPostures, Human Activity Recognition), causal sliding-window features (FLIRT), and the [SensorLM](https://github.com/Google-Health/consumer-health-research/tree/main/sensorlm) sensor–language foundation model. Candidate data sources include the Stanford Wearables set from the DBDP Digital Health Data Repository.
* **Whole Person Physiome (WPP):** a multiorgan knowledge graph (skeletal muscle, bone, tendon, heart, liver, brain, immune tissue) used to model myokine, osteokine and other exerkine cross-talk and to reason about unloading in microgravity, on the Moon (0.16 g) and on Mars (0.38 g).
* **Closed-loop countermeasures:** a bounded policy recalibrates exercise and nutrition recommendations for skeletal muscle and other exerkine organs, spaceflight, health and resilience, and disease. It is **decision support with clinician oversight**, particularly for pregnancy, GDM and cardiovascular conditions. Whether circadian-timed training improves adaptation is evaluated, not assumed.

## Data sources

| Source | Use |
|---|---|
| [MoTrPAC](https://motrpac-data.org) rat endurance-training compendium | whole blood, plasma and 18 solid tissues of male and female rats across 1, 2, 4 and 8 weeks; multiple omes |
| MoTrPAC human acute-exercise data | sedentary adults, endurance and resistance bouts; blood, adipose and muscle |
| HuBMAP | human organ reference atlases and cell-type tables for receptor-side targets |
| Exerkine Atlas | validation of candidate exerkines |
| UniProt, RefSeq | protein and mRNA sequences for exerkines, ligands and receptors (human and rat) |
| DBDP repositories | wearable and CGM data for physiological priors |

MoTrPAC tissues are rat and HuBMAP organs are human, so cross-species results need an explicit ortholog and organ-matching step with an uncertainty estimate. Follow each dataset's data-use terms.

## Tutorials and documentation

* [`docs/exerkine_math_model.md`](docs/exerkine_math_model.md): the mathematical model, with notation, algorithms, a worked numerical example and a list of open issues. Start here to understand the algorithm.
* [`tutorials/exerkine_tutorial.ipynb`](tutorials/exerkine_tutorial.ipynb): step-by-step walkthrough that follows the math reference.
* [`tutorials/exerkine_sequence_generation_gym.ipynb`](tutorials/exerkine_sequence_generation_gym.ipynb): sequence-generation evaluation environment (below).
* Other notebooks in `tutorials/` cover custom single-cell and spatial preprocessing, custom GLM tokenization, and building specialized ligand–receptor databases.

## Exerkine Sequence Generation Gym

A Colab-ready notebook for **constrained variant design** of exerkines, ligands and receptors. It samples variants with ESM-2 masked Gibbs sampling (fixing signal peptides, cysteines and chosen binding residues) and scores each one against a matched random-mutation null:

| Check | Measures |
|---|---|
| Structure | ESMFold pLDDT and backbone RMSD to wild type |
| Novelty | nearest other natural sequence, training-set identity, diversity of the generated set |
| Receptor binding | change in ligand–receptor score vs. wild type, as a percentile of random mutations |
| Rat vs. human | ortholog divergence, where mutations fall, and cross-species binding transfer |

```bash
# open tutorials/exerkine_sequence_generation_gym.ipynb in Google Colab (GPU recommended for ESMFold)
```

These scores are **proxies**. A language model that generates and scores the same sequences is circular, the bundled pair scorer is a demonstration trained on a handful of literature pairs, and none of this replaces experiments. Replace the demo pairs with your Exerkine Atlas pairs and plug in your own protein-level compatibility model.

## Tokenization ablation and one-hot control

Run a matched binary sequence-classification benchmark from an empirical CSV with `sequence` and `label` (`0` or `1`) columns. EXERKINEMAP does not ship this labeled table: it must be assembled from a prespecified sequence-to-phenotype or regulatory task (for example, MoTrPAC-linked differential exercise labels) before benchmarking. The runner uses a stratified train/validation/test split, trains learned vocabularies only on the training sequences, and gives each tokenization arm the same linear-model optimization budget. It reports a position-aware one-hot supervised control alongside character, non-overlapping codon 3-mer, overlapping 6-mer, BPE, unigram and WordPiece tokenization.

```bash
python -m exerkinemap.benchmarks.tokenization_ablation \
  --input /path/to/empirical_sequence_labels.csv \
  --output results/benchmarking/tokenization_ablation.csv
```

The CSV contains per-method accuracy, AUROC, AUPRC, Brier score, feature count and fit time. A same-stem JSON file records the evaluation configuration. These results are empirical outputs; the command does not populate results with placeholder scores.

The default benchmark caps every sequence at 512 nt using a 5-prime (`prefix`) window and caps learned-tokenizer vocabulary training at 25,000 characters. Both settings are recorded in the JSON metadata. Choose a window that matches the prespecified biological task; use `--long-sequence-policy error` to reject rather than truncate long sequences.

### Raw controls

Keep raw negative controls in [sequence_controls.csv](./data/raw/benchmarking/sequence_controls.csv). Each `control_definition` must state whether a control is empirical, technical or reference-derived:

```csv
control_id,species,sequence,refseq_accession,source_url,control_definition
```

Build a benchmark-only processed file by selecting positives explicitly from the RefSeq candidate catalog:

```bash
python -m exerkinemap.benchmarks.sequence_manifest \
  --positive-entity-id FGF19 \
  --positive-entity-id RARRES2 \
  --positive-entity-id LEP \
  --controls data/raw/benchmarking/sequence_controls.csv \
  --candidates data/processed/refseq_sequence_manifest.csv \
  --output data/processed/benchmarks/gdm_sequence_benchmark.csv
```

The builder requires at least three positive records and three raw controls, preserves control provenance, and rejects controls whose sequence duplicates a selected positive. Run tokenization ablation against the resulting processed benchmark file, not the all-positive candidate reference catalog. A benchmark built with reference-derived controls is a technical sequence baseline and must not be presented as evidence of exercise responsiveness or causal biology.

## GDM candidate reference and RefSeq refresh

[gdm_exerkine_candidates.csv](./data/reference/gdm_exerkine_candidates.csv) is a typed catalog of the GDM-network candidates. It includes gene products, receptors, metabolites, reactive species, protein complexes, pathways, extracellular-vesicle cargo and unresolved aliases. Only unambiguous protein-coding genes receive human and rat RefSeq mRNA records; other entities remain in the manifest with blank sequence fields and an explicit `resolution_status`.

Refresh [refseq_sequence_manifest.csv](./data/processed/refseq_sequence_manifest.csv) from NCBI with a contact email:

```bash
python -m exerkinemap.scripts.refresh_refseq_manifest \
  --email d.gomez@me.com
```

The resulting manifest is a candidate reference, not an empirical binary benchmark. Its `label=1` means candidate-list membership; it does not establish exercise responsiveness or causal activity. Do not use it as the tokenization-ablation input until a prespecified empirical negative/control cohort has been added.

## Limitations and responsible use

* **Research use only.** Outputs are not medical advice. Countermeasure and exercise recommendations are bounded decision support that requires clinician oversight.
* **Generated sequences are hypotheses.** Plausibility, predicted structure and predicted binding are proxies; experimental validation is required. Screen generated sequences against toxin and pathogen databases before sharing them.
* **Circular scoring.** Using the same language model to generate and to score sequences overstates quality; use independent predictors.
* **Small labeled set.** A curated exerkine list of roughly 800 entries limits full fine-tuning; prefer adapters and frozen backbones.
* **Species gap.** MoTrPAC tissues are rat and HuBMAP organs are human; transfer needs ortholog mapping and uncertainty estimates.
* **Wearable models.** Sensor–language models trained on consumer-device data must be validated on spaceflight and clinical cohorts before use there.
* **Candidate lists are not evidence.** Membership in the GDM candidate catalog does not establish exercise responsiveness or causal activity.

## License

See the repository license for terms of use.
