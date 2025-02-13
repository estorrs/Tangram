

Wrapper for the Tangram single-cell -> spatial mapping tool.

## Installation
Can be installed via conda with the environment.yml file and pip

```bash
git clone https://github.com/estorrs/Tangram.git
cd Tangram
conda env create --file environment.yml
conda activate tangram
pip install .
```

or used from within a Docker container. See **Docker Usage** section below for more details.

## Usage

Single cell data needs to be in a Scanpy .h5ad object. To convert a Seurat .RDS object to a .h5ad object follow this [example](https://mojaveazure.github.io/seurat-disk/articles/convert-anndata.html) 

By default the wrapper will check if a GPU is available on your machine, if so it will run on said GPU, otherwise it will automatically run on the CPU. Note that if no GPU is available, running with CPU is 1-2 orders of magnitude slower than runtime with a GPU.

General Usage
-------------

usage: run_mapping.py [-h] [--output-dir OUTPUT_DIR] [--label-column LABEL_COLUMN] [--n-variable-genes N_VARIABLE_GENES]
                      [--batch-size BATCH_SIZE] [--invert-y] [--marker-filepath MARKER_FILEPATH]
                      sc_filepath sp_filepath

positional arguments:

  sc_filepath:           Single cell data to be used for label transfer. Should be a .h5ad AnnData object. AnnData should have raw
                        single cell counts in either .X or .raw.X. There should be a column in .obs of name --label-column that
                        contains the annotation to be transfered to the spatial dataset.
                        
  sp_filepath:           Spatial data that is to be mapped. Should be a .h5ad AnnData object or Visium Spaceranger outs. If AnnData
                        object the following information MUST be in the .obs metadata: "x" - x coordinate of voxel, "y" - y
                        coordinate of voxel. .var must coorespond to the gene markers in the dataset (they must match exactly with
                        genes in the single cell dataset to be included in the label transfer). .X holds intensity values or # counts
                        for each voxel. If spaceranger output then no modification is needed.

optional arguments:

  -h, --help            show this help message and exit
  
  --output-dir:
                        Directory in which to save transfer results
                        
  --label-column:
                        Name of column in single cell data storing label to be transfered. Default is "cell_type".
                        
  --n-variable-genes:
                        Number of most variable genes in single cell data to use for each cell type during mapping. Default is 100.
                        
  --batch-size:
                        Number of voxels to map at a time. Helps keep GPU from running out of memory. Default is 3000.
                        
  --invert-y:           Whether to flip y axis in output plots.
  
  --marker-filepath:
                        File containing markers (one gene per line) to use as training genes. If not specified then most variable genes will be extracted from single
                        cell data.

#### Example Usage

Example command where single cell data is stored at inputs/sc.h5ad and cell type label is stored in the column "cell_type". Spaceranger outputs are stored at inputs/visium_outs.

```bash
python tangram/run_mapping.py --output-dir outputs/ --label-column cell_type inputs/sc.h5ad inputs/visium_outs
```


#### Docker Example Usage

The wrapper is available in the following Docker container: `estorrs/tangram_cuda10.2`

Below is an example of running the above **Example Usage** command from within the docker container.

If your system has a GPU and you would like to use it be sure to include the `--gpus all` in the docker command. If your system does not have a GPU the tool will automatically default to CPU.

```bash
docker run --gpus all -v </absolute/filepath/to/input/directory>:/inputs -v </absolute/filepath/to/output/dir>:/outputs -t estorrs/tangram_cuda10.2:0.0.2 python /Tangram/tangram/run_mapping.py --output-dir /outputs/example_outputs --label-column cell_type /inputs/sc.h5ad /inputs/visium_outs
```



## Original tool documentation below:






<img src="https://raw.githubusercontent.com/broadinstitute/Tangram/master/figures/tangram_large.png" width="400"> 

[![PyPI version](https://badge.fury.io/py/tangram-sc.svg)](https://badge.fury.io/py/tangram-sc)

Tangram is a Python package, written in [PyTorch](https://pytorch.org/) and based on [scanpy](https://scanpy.readthedocs.io/en/stable/), for mapping single-cell (or single-nucleus) gene expression data onto spatial gene expression data. The single-cell dataset and the spatial dataset should be collected from the same anatomical region/tissue type, ideally from a biological replicate, and need to share a set of genes. Tangram aligns the single-cell data in space by fitting gene expression on the shared genes. The best way to familiarize yourself with Tangram is to check out [our tutorial](https://github.com/broadinstitute/Tangram/blob/master/tangram_tutorial.ipynb). [![colab tutorial](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SVLUIZR6Da6VUyvX_2RkgVxbPn8f62ge?usp=sharing)

![Tangram_overview](https://raw.githubusercontent.com/broadinstitute/Tangram/master/figures/tangram_overview.png)
Tangram has been tested on various types of transcriptomic data (10Xv3, Smart-seq2 and SHARE-seq for single cell data; MERFISH, Visium, Slide-seq, smFISH and STARmap as spatial data). In our [preprint](https://www.biorxiv.org/content/10.1101/2020.08.29.272831v1), we used Tangram to reveal spatial maps of cell types and gene expression at single cell resolution in the adult mouse brain. More recently, we have applied our method to different tissue types including human lung, human kidney developmental mouse brain and metastatic breast cancer.

***
## Tangram News

* On Jan 28th 2021, Sten Linnarsson gave a [talk](https://www.youtube.com/watch?v=0mxIe2AsSKs) at the WWNDev Forum and demostrated their mappings of the developmental mouse brain using Tangram.

* On Mar 9th 2021, Nicholas Eagles wrote a [blog post](http://research.libd.org/rstatsclub/2021/03/09/lessons-learned-applying-tangram-on-visium-data/#.YFDd7ZNKhax) about applying Tangram on Visium data.



***
## How to run Tangram at cell level

To install Tangram, make sure you have [PyTorch](https://pytorch.org/) and [scanpy](https://scanpy.readthedocs.io/en/stable/) installed. If you need more details on the dependences, look at the `environment.yml` file. 

* install tangram-sc from shell:
```
    pip install tangram-sc
```
* import tangram
```
    import tangram as tg
```

Then load your spatial data and your single cell data (which should be in [AnnData](https://anndata.readthedocs.io/en/latest/anndata.AnnData.html) format), and pre-process them using `tg.pp_adatas`:

```
    ad_sp = sc.read_h5ad(path)
    ad_sc = sc.read_h5ad(path)
    tg.pp_adatas(ad_sc, ad_sp, genes=None)
```

The function `pp_adatas` finds the common genes between adata_sc, adata_sp, and saves them in two `adatas.uns` for mapping and analysis later. Also, it subsets the intersected genes to a set of training genes passed by `genes`. If `genes=None`, Tangram maps using all genes shared by the two datasets. Once the datasets are pre-processed we can map:

```
    ad_map = tg.map_cells_to_space(ad_sc, ad_sp)
```

The returned AnnData,`ad_map`, is a cell-by-voxel structure where `ad_map.X[i, j]` gives the probability for cell $i$ to be in voxel $j$. This structure can be used to project gene expression from the single cell data to space, which is achieved via `tg.project_genes`.

```
    ad_ge = tg.project_genes(ad_map, ad_sc)
```

The returned `ad_ge` is a voxel-by-gene AnnData, similar to spatial data `ad_sp`, but where gene expression has been projected from the single cells. This allows to extend gene throughput, or correct for dropouts, if the single cells have higher quality (or more genes) than single cell data. It can also be used to transfer cell types onto space. 

For more details on how to use Tangram check out [our tutorial](https://github.com/broadinstitute/Tangram/blob/master/tangram_tutorial.ipynb). [![colab tutorial](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SVLUIZR6Da6VUyvX_2RkgVxbPn8f62ge?usp=sharing)

***

## Run Tangram at cluster level

To enable faster training and consume less memory, Tangram mapping can be done at cell cluster level.

Prepare the input data as the same you would do for cell level Tangram mapping. Then map using following code:

```
    ad_map = tg.map_cells_to_space(
                   ad_sc, 
                   ad_sp,         
                   mode='clusters',
                   cluster_label='subclass_label')
```

Provided cluster_label must belong to ad_sc.obs. Above example code is to map at 'subclass_label' level, and the 'subclass_label' is in ad_sc.obs.

To project gene expression to space, use `tg.project_genes` and be sure to set the `cluster_label` argument to the same cluster label in mapping.

```
    ad_ge = tg.project_genes(
                  ad_map, 
                  ad_sc,
                  cluster_label='subclass_label')
```

***

## How Tangram works under the hood
Tangram instantiates a `Mapper` object passing the following arguments:
- _S_: single cell matrix with shape cell-by-gene. Note that genes is the number of training genes.
- _G_: spatial data matrix with shape voxels-by-genes. Voxel can contain multiple cells.

Then, Tangram searches for a mapping matrix _M_, with shape voxels-by-cells, where the element _M\_ij_ signifies the probability of cell _i_ of being in spot _j_. Tangram computes the matrix _M_ by minimizing the following loss:

<img src="https://raw.githubusercontent.com/broadinstitute/Tangram/master/figures/tangram_loss.gif" width="400">

where cos_sim is the cosine similarity. The meaning of the loss function is that gene expression of the mapped single cells should be as similar as possible to the spatial data _G_, under the cosine similarity sense.

The above accounts for basic Tangram usage. In our manuscript, we modified the loss function in several ways so as to add various kinds of prior knowledge, such as number of cell contained in each voxels.

***
## Frequently Asked Questions

#### Do I need a GPU for running Tangram?
A GPU is not required but is recommended. We run most of our mappings on a single P100 which maps ~50k cells in a few minutes. 

#### How do I choose a list of training genes?
A good way to start is to use the top 1k unique marker genes, stratified across cell types, as training genes. Alternatively, you can map using the whole transcriptome. Ideally, training genes should contain high quality signals: if most training genes are rich in dropouts or obtained with bad RNA probes your mapping will not be accurate.

#### Do I need cell segmentation for mapping on Visium data?
You do not need to segment cells in your histology for mapping on spatial transcriptomics data (including Visium and Slide-seq). You need, however, cell segmentation if you wish to deconvolve the data (_ie_ deterministically assign a single cell profile to each cell within a spatial voxel).

#### I run out of memory when I map: what should I do?
Reduce your spatial data in various parts and map each single part. If that is not sufficient, you will need to downsample your single cell data as well.

#### How to use Tangram with Squidpy?
For tutorial, please reference the example [here](https://github.com/broadinstitute/Tangram/blob/master/tutorial_sq_tangram.ipynb). For environment setup, please use squidpy=1.1.0 and reference this [yml file](https://github.com/broadinstitute/Tangram/blob/master/environment.yml).

***
## How to cite Tangram
Tangram has been released in the following publication

Biancalani* T., Scalia* G. et al. - _Deep learning and alignment of spatially-resolved whole transcriptomes of single cells in the mouse brain with Tangram_ [biorXiv 10.1101/2020.08.29.272831](https://www.biorxiv.org/content/10.1101/2020.08.29.272831v3) (2020)

If you have questions, please contact the authors of the method:
- Tommaso Biancalani - <biancalt@gene.com>  
- Gabriele Scalia - <gabriele.scalia@roche.com>

PyPI maintainer:
- Tommaso Biancalani - <biancalt@gene.com>
- Ziqing Lu - <luz21@gene.com>

The artwork has been curated by:
- Anna Hupalowska <ahupalow@broadinstitute.org>
