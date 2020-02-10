# KECA

Implementation of "Knowledge Enhanced Latent Relevance Mining For Question Answering" (ICASSP 2020). Some code in this repository is based on the excellent open-source project https://github.com/wjbianjason/Dynamic-Clip-Attention.

## Requirements

* python 3.x
* Keras 2.2.4
* Tensorflow 1.12.0

## Data

* WikiQA
* TrecQA

## Running

```
usage: main.py [-h] [-t Task] [-m MODEL] [-d HIDDEN_DIM] [-e PATIENCE] [-l LR] [-b BATCH_SIZE] [--no_entity_cmp] [--no_cxt_weight_mixed] [--no_ent_weight_mixed]
```

### WikiQA

* KECA

```
python main.py -t wikiqa
```

* w/o knowledge-aware attention

```
python main.py -t wikiqa --no_ent_weight_mixed --no_cxt_weight_mixed
```

* w/o knowledge-based features

```
python main.py -t wikiqa --no_entity_cmp
```

* w/o external knowledge (KGs)

```
python main.py -t wikiqa --no_entity_cmp --no_cxt_weight_mixed
```

### TrecQA

It's similar to WikiQA, change task from 'wikiqa' to 'trecqa'. For example:

```
python main.py -t trecqa
```
