"""
cast_of_thousands.py
by Tripp Maloney
"""

import argparse
from collections import Counter


def load_texts(tsv):
    authors = []
    texts = []
    with open(tsv, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            fields = line.lower().strip().split("\t")
            authors.append(fields[1])
            texts.append(fields[-1])
        return authors, texts, lines


def over_1k(authors):
    c = Counter(authors)
    keys = list(c.keys())
    vals = list(c.values())

    authors_over_1k = []
    for i in range(len(keys)):
        if vals[i] > 1000:
            authors_over_1k.append(keys[i])

    return authors_over_1k


def thousands(tsv):
    authors, texts, lines = load_texts(tsv)
    authors_over_1k = over_1k(authors)
    print(f"{len(authors_over_1k)} authors with over 1000 emails.")
    print("Retrieving first 1000 emails from each author...")
    thousands = []
    cnt = Counter()
    for i in range(len(texts)):
        if cnt[authors[i]] == 1000:
            continue
        elif authors[i] in authors_over_1k:
            thousands.append(lines[i])
            cnt[authors[i]] += 1


    return thousands


def main(tsv, outfile):
    thous = thousands(tsv)
    with open(outfile, 'w', encoding='utf-8') as out:
        out.write("".join(thous))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='feature vector homework')
    parser.add_argument('--tsv', type=str, default="all_emails_dbl_clean.tsv",
                        help='path to author dataset')
    parser.add_argument('--outfile', default="enron_thousands.tsv",
                        help='output file')
    # parser.add_argument('--punct_path', type=str, default="punct_list.txt",
    #                     help='path to the list of punctuation to use as features')

    args = parser.parse_args()

    main(args.tsv, args.outfile)
