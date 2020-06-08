import argparse
import utils
import pathlib
import collections
from pprint import pprint
from nltk import word_tokenize
from nltk import sent_tokenize
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import random
import tarfile
import os
from newsroom.analyze import Fragments


def read_wcep(path):
    for event in utils.read_jsonl(path):

        if 0:
            articles = [a for a in event['articles'] if a['origin'] == 'WCEP']
            texts = [f'{a["title"]}. {a["text"]}' for a in articles]
        else:
            texts = [f'{a["title"]}. {a["text"]}' for a in event['articles']]
            random.shuffle(texts)
            texts = texts[:100]
        src_sents = [s for text in texts for s in sent_tokenize(text)]
        if len(src_sents) == 0:
            continue
        summary = event['summary']
        tgt_sents = sent_tokenize(summary)
        yield src_sents, tgt_sents


def read_multinews(path):
    indir = pathlib.Path(path)
    sep = 'story_separator_special_tag'
    indir = pathlib.Path(indir)
    src_file = open(indir / 'train.src.txt')
    tgt_file = open(indir / 'train.tgt.txt')
    for src_line, tgt_line in zip(src_file, tgt_file):
        docs = src_line.split(sep)
        src_sents = [s for doc in docs for s in sent_tokenize(doc)]
        tgt_sents = sent_tokenize(tgt_line)
        # print("*" * 100)
        # print("TARGET:")
        # print(tgt_line)
        # print("*"*100)
        yield src_sents, tgt_sents


def read_duc_2004_(root_dir):
    root_dir = pathlib.Path(root_dir)
    docs_dir = root_dir / 'DUC2004_Summarization_Documents/duc2004_testdata/tasks1and2/duc2004_tasks1and2_docs/docs'
    result_dir = root_dir / 'duc2004_results'

    def get_duc_cluster_docs(cluster_id):
        docs = []
        cluster_path = docs_dir / f'd{cluster_id}t'
        for fpath in cluster_path.iterdir():
            with open(fpath) as f:
                raw = f.read()
            text = raw.split("<TEXT>")[1].split("</TEXT>")[0]
            text = " ".join(text.split())
            doc = {
                'fname': fpath.name,
                'cluster_id': cluster_id,
                'text': text
            }
            docs.append(doc)
        docs = sorted(docs, key=lambda x: x['fname'])
        return docs

    cid_to_clusters = {}
    # get reference (models) and peer (participant systems) summaries
    for group in ["models", "peers"]:

        gz_path = result_dir / f'ROUGE/duc2004.task2.ROUGE.{group}.tar.gz'
        tar = tarfile.open(gz_path, "r:gz")
        for member in tar.getmembers():

            author_id = member.name.split(".")[-1]
            cluster_id = member.name.split("/")[-1].split(".")[0].lstrip("D")

            # print(member.name)
            # print('CID:', cluster_id)
            # print()


            with tar.extractfile(member) as f:
                text = str(f.read(), encoding="UTF-8")
            text = " ".join(text.split())

            summary_item = {
                'author_id': author_id,
                'text': text,
                'cluster_id': cluster_id
            }

            if cluster_id not in cid_to_clusters:
                cid_to_clusters[cluster_id] = {
                    'peer_summaries': [],
                    'ref_summaries': [],
                    'id': cluster_id
                }

            if group == "models":
                cid_to_clusters[cluster_id]['ref_summaries'].append(summary_item)
            elif group == "peers":
                cid_to_clusters[cluster_id]['peer_summaries'].append(summary_item)

    # get source documents
    clusters = []
    for cid, c in cid_to_clusters.items():
        docs = get_duc_cluster_docs(cid)
        c['documents'] = docs
        print('CLUSTER:', cid, len(c['documents']))
        clusters.append(c)
    clusters = sorted(clusters, key=lambda x: x['id'])
    print('#clusters:', len(clusters))
    return clusters


def read_duc_2004(path):
    for c in read_duc_2004_(path):
        src_sents = [s for d in c['documents'] for s in sent_tokenize(d['text'])]
        summary = c['ref_summaries'][0]['text']
        tgt_sents = sent_tokenize(summary)
        print(summary)
        yield src_sents, tgt_sents


def read_cnn_dm(path):

    def parse_cnn_dmm_file(text):
        in_sents = []
        out_sents = []
        summary_start = False
        for line in text.split('\n'):
            if line.strip() != '':
                if line == '@highlight':
                    summary_start = True
                else:
                    if summary_start:
                        out_sents.append(line)
                    else:
                        in_sents.append(line)
        return in_sents, out_sents

    indir = pathlib.Path(path)
    for fpath in indir.iterdir():
        text = fpath.read_text()
        in_sents, out_sents = parse_cnn_dmm_file(text)
        yield in_sents, out_sents


def reconstruct_fusion(fragments, a_sents):
    indices=[]
    for f in fragments:
        f_indices = []
        f_ = ' '.join(f)
        for i, s in enumerate(a_sents):
            s_ = ' '.join(word_tokenize(s))
            if f_ in s_:
                f_indices.append(i)
        indices.append(f_indices)
    return indices


def extract_fragments(a_tokens, s_tokens):
    a_size = len(a_tokens)
    s_size = len(s_tokens)
    F = []
    i, j = 0, 0
    # i: for each summary token
    while i < s_size:
        f = []
        # j: for each article token
        while j < a_size:
            # if a&s tokens match:

            if s_tokens[i] == a_tokens[j]:
                i_, j_ = i, j
                # look further until tokens don't match
                while s_tokens[i_] == a_tokens[j_]:
                    i_ += 1
                    j_ += 1
                    if i_ >= s_size or j_ >= a_size:
                        break
                # if new span is larger than previous fragment
                if len(f) < (i_ - i ): # maybe instead: i_ - i - 1
                    f = s_tokens[i: i_] # maybe i_ - 1
                j = j_
            else:
                j += 1
        i += max(len(f), 1)
        j = 0
        if len(f) > 1:
            F.append(f)
    return F


def compute_compression(a_tokens, s_tokens):
    return len(a_tokens) / len(s_tokens)


def compute_density(s_tokens, fragments):
    d = 0
    for frag in fragments:
        d += len(frag)**2
    return d / len(s_tokens)


def compute_coverage(s_tokens, fragments):
    c = 0
    for frag in fragments:
        c += len(frag)
    return c / len(s_tokens)


def make_kde_plots2(results, outpath):
    x = results['coverage']
    y = results['density']
    ax = sns.kdeplot(x, y, cmap="Reds", shade=True, shade_lowest=False)
    ax.set_xlim((-0.2, 1.0))
    ax.set_ylim((-0.2, 5.0))
    plt.savefig(outpath)

    #ax.savefig(outpath)


def make_kde_plots(results, outpath):
    x = results['coverage']
    y = results['density']
    plt.scatter(x, y)
    plt.xlabel('Coverage')
    plt.ylabel('Density')
    plt.savefig(outpath)
    plt.close()


def run(examples, args):
    results = collections.defaultdict(list)
    n = 0
    for i, (a_sents, s_sents) in enumerate(examples):

        if n >= 1000:
            break
        #
        # if i % 10 != 0:
        #     continue

        if i % 100 == 0:
            print(i, n)

        summary = ' '.join(s_sents)
        text = ' '.join(a_sents)
        fragments = Fragments(summary, text)

        coverage = fragments.coverage()
        density = fragments.density()
        compression = fragments.compression()
        #
        # a_tokens = [w for s in a_sents for w in word_tokenize(s)]
        # s_tokens = [w for s in s_sents for w in word_tokenize(s)]
        #
        # if len(s_tokens) == 0 or len(a_tokens) == 0:
        #     continue
        #
        # fragments = extract_fragments(a_tokens, s_tokens)
        # compression = compute_compression(a_tokens, s_tokens)
        # density = compute_density(s_tokens, fragments)
        # coverage = compute_coverage(s_tokens, fragments)
        #
        # if density > 0:
        #     density = density / len(s_tokens)

        #
        # print("frags", len(fragments))
        # print('COV', coverage, 'DEN', density, 'COMP', compression)
        #
        # for f in fragments:
        #     print(f)
        # print()

        #
        # if coverage == 0:
        #     print('coverage:', coverage)
        #
        #     print('*** S ***')
        #     for s in s_sents:
        #         print(s)
        #
        #     print()
        #     print('*** A ***')
        #     for s in a_sents[:5]:
        #         print(s)
        #
        #     print()
        #     print()
        #
        # print()
        # print('*** FRAGMENTS ***:')
        # for f in fragments:
        #     print(' '.join(f))
        # print()
        #
        print('compression:', compression)
        print('density:', density)
        print('coverage:', coverage)
        print('='*100)

        results['compression'].append(compression)
        results['density'].append(density)
        results['coverage'].append(coverage)
        n += 1

    utils.writejson(results, args.o)
    #make_kde_plots2(results, args.o + '/kde.png')


def main(args):
    examples = []
    if args.corpus == 'cnn-dm':
        examples = read_cnn_dm(args.i)
    elif args.corpus == 'multinews':
        examples = read_multinews(args.i)
    elif args.corpus == 'wcep':
        examples = read_wcep(args.i)
    elif args.corpus == 'duc':
        examples = read_duc_2004(args.i)
    run(examples, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--i', required=True)
    parser.add_argument('--o', required=True)
    parser.add_argument('--corpus', default='wcep')
    main(parser.parse_args())
