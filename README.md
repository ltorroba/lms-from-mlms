# Deriving Language Models from Masked Language Models

This is the code for the paper [Deriving Language Models from Masked Language Models (Torroba Hennigen and Kim, ACL 2023)](https://aclanthology.org/2023.acl-short.99/).

## Setup

First, initialize the environment using `conda env create -f environment.yml`

After installing packages, open up a python shell and run:
```
>>> import nltk
>>> nltk.download('punkt')
```

## Reproducing results

### Compute compatibility of MLMs

To obtain the compatibility of the different MLMs (finetuned and pretrained), just run the metrics of pairwise coherent models below. That file will automatically include compatibility metrics.


### Compute metrics of pairwise coherent models

To compute the NLL, JSD, etc. of the different pairwise coherency models, run _(NOTE: These take a while to run, around ~2 days per script)_:

```
MODEL={bert-base-cased,bert-large-cased} DATASET={snli,xsum} BLOCK_CONTIGUOUS={true,false} ./scripts/evaluate_kl_compatibility.sh
```

To run manually, run:

```
python evaluate_kl_compatibility.py bert-base-cased --dataset snli --schemes naive mrf mrf-local hcb-both iter --use-gpu --batch-size 256 --max-datapoints 100 --joint-size 50
```

### Plots

To generate plots of relationship between (syntactic) distance and NLL, run
```
python print_ppl_examples.py results/kl-compatibility/bert-base-cased.snli.1000.pkl naive mrf mrf-local hcb-both iter bert-base-cased --graph-mode pnll --distance distance --average
```

### Appendix D

The code for reproducing the example in Appendix D is in `evaluate_mrf_pathological_sentences.py`

## Additional information

#### Citation

If you found the paper or code useful, you can cite us as follows:

```
@inproceedings{torroba-hennigen-kim-2023-deriving,
    title = "Deriving Language Models from Masked Language Models",
    author = "Torroba Hennigen, Lucas  and
      Kim, Yoon",
    editor = "Rogers, Anna  and
      Boyd-Graber, Jordan  and
      Okazaki, Naoaki",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-short.99",
    doi = "10.18653/v1/2023.acl-short.99",
    pages = "1149--1159",
}
```

#### Problems

To ask questions or report problems, please open an [issue](https://github.com/ltorroba/lms-from-mlms/issues).
