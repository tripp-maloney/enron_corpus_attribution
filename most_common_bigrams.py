"""
most_common_bigrams.py
By Tripp Maloney, Georgetown MLC '19
For final project for LING 402, Forensic Linguistics
Last edited 3 May 2019
"""

import argparse
from collections import Counter


def main(tsv):
    # reads a tsv of texts for authorship attribution with texts in the final column
    # Writes a list of the 1000 most common bigrams
    # Text encoding chosen for Arabic compatibility

    texts = []
    with open(tsv, 'r', encoding='windows-1256') as f:
        lines = f.readlines()
        for line in lines:
            fields = line.strip().split("\t")
            texts.append(fields[-1])
    print("Fetching unigrams...\n")
    unigrams = []
    for text in texts:
        t_unigrams = text.split(" ")
        unigrams += t_unigrams
    print("purging empties...")
    unigrams = [unigram for unigram in unigrams if unigram != ""]
    print("Fetching bigrams...\n")
    bigrams = []
    for i in range(len(unigrams) - 1):
        bigrams.append((unigrams[i].lower(), unigrams[i + 1].lower()))

    print(f"{str(len(set(bigrams)))} unique bigrams!\n")

    print("Fetching 1000 most common bigrams...\n")

    c = Counter(bigrams)
    top_1k = c.most_common(1000)
    print(f"Done! top bigram frequency = {str(top_1k[0][1])}, 1000th bigram frequency = {str(top_1k[999][1])}")
    t1k = [i[0] for i in top_1k]
    print("writing top list to 'top_bigrams.tsv'")
    with open("top_bigrams.tsv", "w", encoding='windows-1256') as out:
        for x in t1k:
            out.write(x[0] + "\t" + x[1] + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--texts", default="all_emails_dbl_clean.tsv",
                        help="tsv of texts, this script will find the 1000 most common bigrams")
    args = parser.parse_args()

    main(args.texts)
