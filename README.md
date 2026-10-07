# EXERKINEMAP

EXERKINEMAP (EXERcise KINEmatics Multiomics single-cell Analysis and spatial omics holistic modeling and maPping) is a computational framework for modeling how exercise-responsive molecular signals move between cells, tissues, and organs. It combines genomic and protein language models, single-cell and spatial omics, and wearable telemetry in a stateful virtual cell world model: an encoder, a transition operator, and a decoder, with a causal-set structure that restricts every prediction to its causal past. The model grows along two axes: a virtual cell bank across cell types, genotypes, and conditions (healthy control, GDM, T2D, CVD, obesity), and a composition from cells to tissues, organs, and the whole person, including the fetal–maternal interface. A forward map predicts the signaling response to an intervention, and an inverse map proposes exerkine sequences that move cell states toward a healthy reference.

## Application within EXERKINEMAP
EXERKINEMAP leverages this omics FM framework to construct a stateful virtual cell world model for exercise-responsive signaling. By utilizing foundation models as the encoder and transition operator, EXERKINEMAP translates raw nucleotide sequences and amino-acid representations into predictive ligand-receptor communication networks.

To map the causal-set propagation from the cellular level to the whole-person physiome, EXERKINEMAP integrates multi-modal datasets directly into the fine-tuning and validation stages. Sequence generation tasks for exercise response targets bypass older model repositories and focus exclusively on high-resolution data from the Molecular Transducers of Physical Activity Consortium (MoTrPAC). This ensures that the generated exerkine sequences and spatial omics alignments are grounded in the most robust, temporally resolved physiological datasets available.

Suggested topics: bioinformatics, single-cell, spatial-omics, foundation-models, genomic-language-model, protein-language-model, virtual-cell, world-model, exercise-physiology, multi-omics
By integrating genomic/protein language models with single-cell and spatial omics, EXERKINEMAP allows you to input target molecular sequences to identify and map candidate exerkines. It is heavily optimized for exploring exercise-responsive physiological adaptations—such as evaluating exerkine signaling for Gestational Diabetes Mellitus (GDM)—using multi-modal datasets from [MoTrPAC (rat and human)](https://motrpac-data.org).

## The Pipeline

* **ExerkineRNA (GLM):** Tokenizes nucleotide sequences (codons, regulatory tags) into contextual genomic embeddings.
* **ExerkineProtein (PLM):** Generates and refines amino-acid representations using models like ESM-2.
* **ExerkineMap:** Aligns these sequence embeddings with spatial and single-cell omics to construct predictive ligand-receptor communication networks.

## Quick Start

Map your target sequences and generate exerkine communication networks in a few steps.

### 1. Install Dependencies

```bash
conda create -n exerkinemap 
conda activate exerkinemap
pip install --upgrade pip
pip install -r requirements.txt

```

### 2. Prepare Your Sequences

Place your target RNA or protein sequences into `data/raw/sequences/`. Ensure your MoTrPAC reference metadata is situated in `data/raw/metadata/`.

### 3. Run the Model

Execute the preprocessing and mapping workflow to align your target sequences against the multi-omics data:

```bash
# Prepare references and build sequence embeddings
python workflows/01_download_data.py
python workflows/04_build_sequence_reference.py

# Execute the mapping pipeline
python run_exerkinemap.py --input data/raw/sequences/ --output results/

```

## Tutorials & Advanced Usage

For custom single-cell/spatial preprocessing, custom GLM tokenization, or building specialized ligand-receptor databases, check out the notebooks in the `tutorials/` directory.

## Tokenization Ablation and One-Hot Control

Run a matched binary sequence-classification benchmark from an empirical CSV with `sequence` and `label` (`0` or `1`) columns. EXERKINEMAP does not ship this labeled table: it must be assembled from a prespecified sequence-to-phenotype or regulatory task (for example, MoTrPAC-linked differential exercise labels) before benchmarking. The runner uses a stratified train/validation/test split, trains learned vocabularies only on the training sequences, and gives each tokenization arm the same linear-model optimization budget. It reports a position-aware one-hot supervised control alongside character, non-overlapping codon 3-mer, overlapping 6-mer, BPE, unigram, and WordPiece tokenization.

```bash
python -m exerkinemap.benchmarks.tokenization_ablation \
  --input /path/to/empirical_sequence_labels.csv \
  --output results/benchmarking/tokenization_ablation.csv
```

The CSV contains per-method accuracy, AUROC, AUPRC, Brier score, feature count, and fit time. A same-stem JSON file records the evaluation configuration. These results are empirical outputs; the command does not populate results with placeholder scores.

The default benchmark caps every sequence at 512 nt using a 5-prime (`prefix`) window and caps learned-tokenizer vocabulary training at 25,000 characters. Both settings are recorded in the JSON metadata. Choose a window that matches the prespecified biological task; use `--long-sequence-policy error` to reject rather than truncate long sequences.

### Raw controls

Keep raw negative controls in [sequence_controls.csv](./data/raw/benchmarking/sequence_controls.csv). Each `control_definition` must state whether a control is empirical, technical, or reference-derived:

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

## GDM Candidate Reference and RefSeq Refresh

[gdm_exerkine_candidates.csv](./data/reference/gdm_exerkine_candidates.csv) is a typed catalog of the GDM-network candidates. It includes gene products, receptors, metabolites, reactive species, protein complexes, pathways, extracellular-vesicle cargo, and unresolved aliases. Only unambiguous protein-coding genes receive human and rat RefSeq mRNA records; other entities remain in the manifest with blank sequence fields and an explicit `resolution_status`.

Refresh [refseq_sequence_manifest.csv](./data/processed/refseq_sequence_manifest.csv) from NCBI with a contact email:

```bash
python -m exerkinemap.scripts.refresh_refseq_manifest \
  --email you@example.org
```

The resulting manifest is a candidate reference, not an empirical binary benchmark. Its `label=1` means candidate-list membership; it does not establish exercise responsiveness or causal activity. Do not use it as the tokenization-ablation input until a prespecified empirical negative/control cohort has been added.

## License

See the repository license for terms of use.
