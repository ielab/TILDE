# TILDEv2
This is the official repository for our paper [Fast Passage Re-ranking with Contextualized Exact Term
Matching and Efficient Passage Expansion.](https://arxiv.org/pdf/2108.08513)

## Passage re-ranking with TILDEv2
Unlike original TILDE which pre-computes term weights over all the BERT vocabulary, TILDEv2 only computes term weights with expanded passages. This requires to only store in the index the score of terms that appear in the expanded passages (rather than all the vocabulary), thus producing indexes that are 99% smaller than those of TILDE.

To try out passage ranking with TILDEv2 on MS MARCO passage ranking task, you need to first expand the whole MS MARCO passage collection. We adapt TILDE as a passage expansion model to do the job. Please following the instructions in [TILDE READEME](../README.md/#passage-expansion-with-tilde). Note, the TILDEv2 model used in this example are trained with TILDE expansion with `m=200` (see section5.4 in our [paper](https://arxiv.org/pdf/2108.08513)). Hence make sure you set `--topk 200` for TILDE passage expension.

### Indexing the collection

First, run the following command to index the whole MS MARCO expanded collection:

```
python3 indexingv2.py \
--ckpt_path_or_name ielab/TILDEv2-TILDE200-exp \
--collection_path path/to/collection/expanded/ 
```
If you have a gpu with big memory, you can set `--batch_size` that suits your gpu the best.

This command will create a `.hdf5` file that stores contextulized term weights of expanded MS MARCO passages and a `.npy` file that stores document ids in the folder `./data/index/TILDEv2`. Compare to TILDE index file which is more than 500GB, TILDEv2 only requries around 4GB.


### Re-rank BM25 results.
After you got the index, now you can use TILDEv2 to re-rank BM25 results:

```
python3 inferencev2.py \
--index_path ./data/index/TILDEv2 \
--query_path ./data/queries/DL2019-queries.tsv \
--run_path ./data/runs/run.trec2019-bm25.res \
--save_path ./data/runs/TILDEv2.txt
```
It will generate another run file in `./data/runs/` and also will print the query latency of the average query processing time and re-ranking time:

```
Query processing time: 0.1 ms
passage re-ranking time: 20.5 ms
```
In our case, we use an intel cpu version of Mac mini without cuda library, this means we do not use any gpu in this example. TILDEv2 only uses 0.1ms to compute the query sparse representation and 20.5ms to re-rank 1000 passages retrieved by BM25. If use smaller `--topk` for passage expansion the query latency will be smaller. In our paper, we reproted latency of docTquery-T5 passage expansion which latency is similar to `--topk 128`. 

We note, the direct index in the `inferencev2.py` script is implemented with python built-in dictionary, which is memory inefficient (requires around 40G memory to build the index for the whole collection). If you don't have big enough memory, you can try to use `inferencev2_memory_efficient.py` with the same configs. This inference code creates direct posting lists 'on-the-fly', but of couse has higher query latency.

Now let's evaluate the TILDEv2 run:

```
trec_eval -m ndcg_cut.10 -m map ./data/qrels/2019qrels-pass.txt ./data/runs/TILDEv2.txt
```
we get:

```
map                     all     0.4595
ndcg_cut_10             all     0.6747
```
This means, with only 0.1ms + 20.5ms add on BM25, TILDEv2 can improve the performance quite a bit. We note the ndcg@10 score is improved by ~16% over the original TILDE.


## To train TILDEv2
To be available soon