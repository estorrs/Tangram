import argparse
import os
from importlib import reload
from pathlib import Path

import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import anndata
import numpy as np
import pandas as pd

import torch
import tangram as tg

plt.rcParams['figure.dpi'] = 300

parser = argparse.ArgumentParser()

parser.add_argument('sc_filepath', type=str,
                    help='Single cell data to be used for label transfer. \
Should be a .h5ad AnnData object. AnnData should have raw single cell counts \
in either .X or .raw.X. There should be a column in .obs of name --label-column \
that contains the annotation to be transfered to the spatial dataset.')

parser.add_argument('sp_filepath', type=str,
                    help='Spatial data that is to be mapped. \
Should be a .h5ad AnnData object or Visium Spaceranger outs. \
If AnnData object the following information MUST be in the .obs \
metadata: "x" - x coordinate of voxel, "y" - y coordinate of voxel. \
.var must coorespond to the gene markers in the dataset (they must match \
exactly with genes in the single cell dataset to be included in the label \
transfer). .X holds intensity values or # counts for each voxel. \
If spaceranger output then no modification is needed.')

parser.add_argument('--output-dir', type=str, default='outputs',
                    help='Directory in which to save transfer results')

parser.add_argument('--label-column', type=str, default='cell_type',
                    help='Name of column in single cell data storing label \
to be transfered.')

parser.add_argument('--n-variable-genes', type=int, default=100,
                    help='Number of most variable genes in single cell data \
to use for each cell type during mapping.')

parser.add_argument('--batch-size', type=int, default=3000,
                    help='Number of voxels to map at a time. Helps keep \
GPU from running out of memory.')

parser.add_argument('--invert-y', default=False, action='store_true',
                    help='Whether to flip y axis in output plots')

parser.add_argument('--marker-filepath',
                    help='Markers to use as training genes. If not \
specified then most variable genes will be extracted from single cell \
data.')


def preprocess_spatial(a):
    a.obs['y'] = a.obs['array_col'].to_list()
    a.obs['x'] = a.obs['array_row'].to_list()
    return a


def get_spatial(fp):
    ad_sp = sc.read_visium(fp)
    ad_sp.var_names_make_unique()
    ad_sp = preprocess_spatial(ad_sp)
    return ad_sp


def is_raw_counts(X):
    if 'sparse' in str(type(X)):
        return all([float(x).is_integer()
                    for x in X[0].toarray().flatten()])
    return all([float(x).is_integer()
                for x in X[0].flatten()])


def reformat_sc(a):
    # see where raw counts are
    in_X = is_raw_counts(a.X)

    if not in_X and a.raw is None:
        raise RuntimeError('Raw count data not found in .x or .raw.X in \
single cell data')

    if not in_X:
        in_raw = is_raw_counts(a.raw.X)

    if not in_X and not in_raw:
        raise RuntimeError('Raw count data not found in .X or .raw.X \
in single cell data')

    if not in_X:
        a = anndata.AnnData(X=a.raw.X, obs=a.obs, var=a.var, obsm=a.obsm)

    return a


def get_sc(fp, label='cell_type'):
    a = sc.read_h5ad(fp)
    a.var_names_make_unique()
    # make sure no repeat genes
    if len(set(a.var.index)) != a.shape[1]:
        # just drop the duplicate for speed reasons if needed
        a = a[:, ~a.var.index.duplicated()]

    a = reformat_sc(a)

    sc.pp.normalize_total(a)
    a.obs['subclass_label'] = a.obs[label].to_list()

    return a


def get_markers(ad_sc, n=100):
    markers = set()
    for ct in sorted(set(ad_sc.obs['subclass_label'])):
        markers.update(ad_sc.uns['rank_genes_groups']['names'][ct][:n])
    markers = sorted(markers)
    return markers


def map_data(ad_sp, ad_sc, markers, b=3000):
    ad_map_d, sp_map_d = {}, {}
    for i in range(0, ad_sc.shape[0], b):
        sp = ad_sp[i:i+b].copy()
        if sp.shape[0]:
            tg.pp_adatas(ad_sc, sp, genes=markers)
            ad_map_d[i] = tg.map_cells_to_space(
                adata_sc=ad_sc,
                adata_sp=sp,
                device=f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu',
            )
            sp_map_d[i] = sp
            print(f'{i + b} voxels mapped')

    ad_sp = anndata.concat(sp_map_d.values())
    ad_sp.var = sp_map_d[0].var.copy()
    train_genes_df = ad_map_d[0].uns['train_genes_df']
    training_history = ad_map_d[0].uns['training_history']

    for a in ad_map_d.values():
        a.uns.pop('train_genes_df')
        a.uns.pop('training_history')

    ad_map = anndata.concat(ad_map_d.values(), axis=1)
    ad_map.uns['train_genes_df'] = train_genes_df
    ad_map.uns['training_history'] = training_history
    ad_map.obs = ad_map_d[0].obs.copy()

    return ad_map, ad_sp


def main(args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    ext = args.sp_filepath.split('.')[-1]
    if ext == 'h5ad':
        ad_sp = sc.read_h5ad(args.sp_filepath)
    else:
        ad_sp = get_spatial(args.sp_filepath)

    print('loading single cell data')
    ad_sc = get_sc(args.sc_filepath, label=args.label_column)

    if args.marker_filepath is None:
        print('extracting most variable genes')
        sc.tl.rank_genes_groups(ad_sc, groupby='subclass_label')
        markers = get_markers(ad_sc, n=args.n_variable_genes)
    else:
        markers = sorted(
            pd.read_csv(args.marker_filepath).values[:, 0].flatten())

    print('mapping datasets')
    ad_map, ad_sp = map_data(ad_sp, ad_sc, markers, b=args.batch_size)

    tg.plot_cell_annotation(ad_map, ad_sp, x='x', y='y',
                            annotation='subclass_label', nrows=5, ncols=4,
                            robust=True, perc=0.05, invert_y=args.invert_y)
    plt.savefig(os.path.join(args.output_dir, f'label_probabilities.png'))
    plt.clf()

    ad_sp.obs['tangram_cell_type'] = [
        ad_sp.obsm['tangram_ct_pred'].columns[i]
        for i in np.argmax(ad_sp.obsm['tangram_ct_pred'].values, axis=1).flatten()]
    ad_sp.obsm['tangram_ct_pred'].to_csv(
        os.path.join(args.output_dir, f'prediction_probabilities.txt'),
        sep='\t')

    ax = sns.scatterplot(data=ad_sp.obs, x='x', y='y', hue='tangram_cell_type',
                         linewidth=0, s=.6)
    if args.invert_y:
        ax.invert_yaxis()
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(os.path.join(args.output_dir, f'label_predictions.png'))
    plt.clf()

    tg.plot_training_scores(ad_map, bins=10, alpha=.5)
    plt.savefig(os.path.join(args.output_dir, f'training_scores.png'))
    plt.clf()

    ad_ge = tg.project_genes(adata_map=ad_map, adata_sc=ad_sc)

    try:
        df_all_genes = tg.compare_spatial_geneexp(ad_ge, ad_sp, ad_sc)
        tg.plot_test_scores(df_all_genes)
        plt.savefig(os.path.join(args.output_dir, f'test_scores.png'))
        plt.clf()
    except:
        print('Error evaluating test genes. Continuing')

    genes = ['cd3g', 'cd4', 'cd8a', 'cd68', 'ms4a1', 'lag3', 'foxp3', 'ca9',
             'cdh1', 'epcam', 'krt18', 'pecam1', 'bgn']
    tg.plot_genes(genes, adata_measured=ad_sp, adata_predicted=ad_ge,
                  robust=True, perc=0.02, x='x', y='y', invert_y=args.invert_y)
    plt.savefig(os.path.join(args.output_dir, f'gene_probabilities.png'))
    plt.clf()

    ad_sp.write_h5ad(os.path.join(args.output_dir, f'sp.h5ad'))
    ad_ge.write_h5ad(os.path.join(args.output_dir, f'ge.h5ad'))

    print(f'mapping complete: output saved to {args.output_dir}')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
