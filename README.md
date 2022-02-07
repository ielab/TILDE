# TILDE
This is the official repository for the SIGIR2021 paper [TILDE: Term Independent Likelihood moDEl for Passage Re-ranking](http://ielab.io/publications/arvin-2021-TILDE).

And the official repository for our arxiv paper [Fast Passage Re-ranking with Contextualized Exact Term
Matching and Efficient Passage Expansion.](https://arxiv.org/pdf/2108.08513)


TILDE now is on huggingface model hub. You can directly download and use it by typing in your Python code:

```
from transformers import BertLMHeadModel, BertTokenizerFast

model = BertLMHeadModel.from_pretrained("ielab/TILDE")
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
```
As you see, TILDE is a `BertLMHeadModel`, you may get a warning from `transformers` that says:

```
If you want to use `BertLMHeadModel` as a standalone, add `is_decoder=True`.
```
Please ignore this warning, because we indeed will use TILDE as a standalone but still treat it as a transformer encoder.

## Updates
- 13/09/2021 Release the reproducing of uniCOIL with [TILDE passage expansion](#passage-expansion-with-tilde).
- 17/09/2021 Release the code for [TILDE passage expansion](#passage-expansion-with-tilde).
- 02/10/2021 Release the code for [inferencing TILDEv2](TILDEv2).
- 04/10/2021 Release the code for [training TILDE](#to-train-tilde).
- 31/10/2021 Release the code for [training TILDEv2](TILDEv2/README.md/#to-train-tildev2).


## Prepare environment and data folder
To train and inference TILDE, we use python3.7, the [huggingface](https://huggingface.co/) implementation of BERT and [pytorch-lightning](https://www.pytorchlightning.ai/). 

Run `pip install -r requirements.txt` in the root folder to set up the libraries that will be used by this repository.

To repoduce the results presented in the paper, you need to download `collection.tar.gz` from the MS MARCO passage ranking repository; this is available at this [link](https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz). Unzip and move `collection.tsv` into the folder `./data/collection`.

In order to reproduce the results with minimum effort, we also provided the TREC DL2019 query file (`DL2019-queries.tsv`) in the `./data/queries/` folder, and its qrel file (`2019qrels-pass.txt`) in `./data/qrels/`. There is also a TREC style BM25 run file (`run.trec2019-bm25.res`) generated by [pyserini](https://github.com/castorini/pyserini) in `./data/runs/` folder which we will use to re-rank.

## Passage re-ranking with TILDE
TILDE uses BERT to pre-compute passage representations. Since the MS MARCO passage collection has around 8.8m passages, it will require more than 500G to store the representations of the whole collection. To quickly try out TILDE, in this example, we only pre-compute passages that we need to re-rank.

### Indexing the collection

First, run the following command from the root:

```
python3 indexing.py \
--ckpt_path_or_name ielab/TILDE \
--run_path ./data/runs/run.trec2019-bm25.res \
--collection_path path/to/collection.tsv \
--output_path ./data/index/TILDE
```
If you have a gpu with big memory, you can set `--batch_size` that suits your gpu the best.

This command will create a mini index in the folder `./data/index/TILDE` that stores representations of passages in the BM25 run file.

If you want to index the whole collection, simply run:

```
python3 indexing.py \
--ckpt_path_or_name ielab/TILDE \
--collection_path path/to/collection.tsv \
--output_path ./data/index/TILDE
```
### Re-rank BM25 results.
After you got the index, now you can use TILDE to re-rank BM25 results.

Let‘s first check out what is the BM25 performance on TREC DL2019 with [trec_eval](https://github.com/usnistgov/trec_eval):

```
trec_eval -m ndcg_cut.10 -m map ./data/qrels/2019qrels-pass.txt ./data/runs/run.trec2019-bm25.res
```
we get:

```
map                     all     0.3766
ndcg_cut_10             all     0.4973
```

Now run the command bellow to use TILDE to re-rank BM25 top1000 results:

```
python3 inference.py \
--run_path ./data/runs/run.trec2019-bm25.res \
--query_path ./data/queries/DL2019-queries.tsv \
--index_path ./data/index/TILDE/passage_embeddings.pkl \
--save_path ./data/runs/TILDE_alpha1.txt
```
It will generate another run file in `./data/runs/` and also will print the query latency of the average query processing time and re-ranking time:

```
Query processing time: 0.2 ms
passage re-ranking time: 6.7 ms
```
In our case, we use an intel cpu version of Mac mini without cuda library, this means we do not use any gpu in this example. TILDE only uses 0.2ms to compute the query sparse representation and 6.7ms to re-rank 1000 passages retrieved by BM25. Note, by default, the code uses a pure query likelihood ranking setting (alpha=1).

Now let's evaluate the TILDE run:

```
trec_eval -m ndcg_cut.10 -m map ./data/qrels/2019qrels-pass.txt ./data/runs/TILDE_alpha1.txt
```
we get:

```
map                     all     0.4058
ndcg_cut_10             all     0.5791
```
This means, with only 0.2ms + 6.7ms add on BM25, TILDE can improve the performance quite a bit. If you want more improvement, you can interpolate query likelihood score with document likelihood by:

```
python3 inference.py \
--run_path ./data/runs/run.trec2019-bm25.res \
--query_path ./data/queries/DL2019-queries.tsv \
--index_path ./data/index/TILDE/passage_embeddings.pkl \
--alpha 0.5 \
--save_path ./data/runs/TILDE_alpha0.5.txt
```
you will get higher query latency:

```
Query processing time: 68.0 ms
passage re-ranking time: 16.4 ms
```
This is because now TILDE has an extra step of using BERT to compute query dense representation. As a trade-off you will get higher effectiveness:

```
trec_eval -m ndcg_cut.10 -m map ./data/qrels/2019qrels-pass.txt ./data/runs/TILDE_alpha0.5.res 
```
```
map                     all     0.4204
ndcg_cut_10             all     0.6088
```
## Passage expansion with TILDE
In addition to the passage reranking model, TILDE can also serve as a passage expansion model. Our paper "[Fast Passage Re-ranking with Contextualized Exact Term
Matching and Efficient Passage Expansion](https://arxiv.org/pdf/2108.08513)" describes the algorithm of using TILDE to do passage expansion. Here, we give the example of expanding the MS MARCO passage collection with TILDE. 

First, make sure you have downloaded [collection.tsv](https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz) and unzipped it in `data/collection/`. Then just need to run the following command:

```
python3 expansion.py \
--corpus_path path/to/collection.tsv \
--output_dir ./data/collection/expanded
--topk 200
```
This python script will generate a jsonl file that contains expanded passages in `data/collection/` as well. Each line in the file has a pid and its corresponding expanded passage:

```
{"pid": str, "psg": List[int]}
```
This takes around 7 hours to expand the whole MS MARCO passage collection on a single tesla v100 GPU. Note, by default, we store the token ids. You can also store the raw text of expanded passages by adding the flag `--store_raw`. This means the format becomes `{"pid": str, "psg": str}`. Also note, `--store_raw` will slow down the speed a little bit.

For impact of `--topk`, we refere to the experiments described in our [paper](https://arxiv.org/pdf/2108.08513) (section 5.4).


- To reproduce the uniCOIL results with TILDE passage expansion, we refer to [pyserini](https://github.com/castorini/pyserini/blob/master/docs/experiments-unicoil-tilde-expansion.md) and [anserini](https://github.com/castorini/anserini/blob/master/docs/experiments-msmarco-passage-unicoil-tilde-expansion.md) instructions.

- To reproduce TILDEv2 results with TILDE passage expansion, check out the instructions in[`/TILDEv2`](TILDEv2) folder.

## To train TILDE
We use the same training data as used for training [docTTTTTquery](https://github.com/castorini/docTTTTTquery) where each line in the dataset are relevant document-query pair separated by `/t`.

Frist, download the training data (`doc_query_pairs.train.tsv`) from the original docTTTTTquery [repo](https://www.dropbox.com/s/5i64irveqvvegey/doc_query_pairs.train.tsv?dl=1). This dataset contains approximately 500,000 passage-query pairs used to train the model.

After you downloaded the training dataset, then simply run the following command to kick off the training:

```
python3 train_tilde.py \
--train_path path/to/doc_query_pairs.train.tsv \
--save_path tilde_ckpts \
--gradient_checkpoint
```

Note, we use `--gradient_checkpoint` flag to trade off training speed for larger batch size, if you have GPU with big memory, consider removing this flag for faster training. Pytorch-lightning model checkpoints will be saved after each epoch and the final checkpoint will be converted and saved as a Huggingface model.



