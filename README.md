# EXERKINEMAP

**EXERKINEMAP** is a multimodal computational framework for mapping exercise-responsive molecular signaling and intercellular crosstalk.

By integrating genomic/protein language models with single-cell and spatial omics, EXERKINEMAP allows you to input target molecular sequences to identify and map candidate exerkines. It is heavily optimized for exploring exercise-responsive physiological adaptations—such as evaluating exerkine signaling for Gestational Diabetes Mellitus (GDM)—using multi-modal datasets from [MoTrPAC (rat and human)](https://motrpac-data.org).

## The Pipeline

* **ExerkineRNA (GLM):** Tokenizes nucleotide sequences (codons, regulatory tags) into contextual genomic embeddings.
* **ExerkineProtein (PLM):** Generates and refines amino-acid representations using models like ESM-2.
* **ExerkineMap:** Aligns these sequence embeddings with spatial and single-cell omics to construct predictive ligand-receptor communication networks.

## Quick Start

Map your target sequences and generate exerkine communication networks in a few steps.

### 1. Install Dependencies

```bash
conda create -n exerkinemap python=3.11 -y
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

## License

See the repository license for terms of use.
