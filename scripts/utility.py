from tqdm import tqdm
from nltk.corpus import stopwords
import re


def load_run(run_path, run_type='trec'):
    run = {}
    with open(run_path, 'r') as f:
        for line in tqdm(f, desc="loading run...."):
            if run_type == 'msmarco':
                qid, docid, score = line.strip().split("\t")
            elif run_type == 'trec':
                qid, _, docid, rank, score, _ = line.strip().split(" ")
            qid = int(qid)
            docid = int(docid)
            if qid not in run.keys():
                run[qid] = []
            run[qid].append(docid)
    return run


def load_collection(collection_path):
    collection = {}
    with open(collection_path, 'r') as f:
        for line in tqdm(f, desc="loading collection...."):
            docid, text = line.strip().split("\t")
            collection[int(docid)] = text
    return collection


def load_queries(query_path):
    query = {}
    with open(query_path, 'r') as f:
        for line in tqdm(f, desc="loading query...."):
            qid, text = line.strip().split("\t")
            query[int(qid)] = text
    return query


def get_batch_text(start, end, docids, collection):
    batch_text = []
    for docid in docids[start: end]:
        batch_text.append(collection[docid])
    return batch_text


def clean_vacab(tokenizer, do_stopwords=True):
    if do_stopwords:
        stop_words = set(stopwords.words('english'))
        # keep some common words in ms marco questions
        stop_words.difference_update(["where", "how", "what", "when", "which", "why", "who"])

    vocab = tokenizer.get_vocab()
    tokens = vocab.keys()

    # good_token = []
    good_ids = []
    # bad_token = []
    bad_ids = []

    for stop_word in stop_words:
        ids = tokenizer(stop_word, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            # bad_token.append(stop_word)
            bad_ids.append(ids[0])

    for token in tokens:
        token_id = vocab[token]
        if token_id in bad_ids:
            continue

        if token[0] == '#' and len(token) > 1:
            # bad_token.append(token)
            good_ids.append(token_id)
        else:
            if not re.match("^[A-Za-z0-9_-]*$", token):
                # bad_token.append(token)
                bad_ids.append(token_id)
            else:
                # good_token.append(token)
                good_ids.append(token_id)

    return good_ids, bad_ids
