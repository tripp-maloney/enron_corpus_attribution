"""
method_evaluator.py
By Tripp Maloney
Last edited 20190423

This script is meant to evaluate different computational authorship attribution methods on the Enron email dataset.
The intended input for this script is a comprehensive .tsv containing of the following format:

email#  authname    email-text
001537	arnold-j	how was the concert?
"""

import numpy as np
import argparse
from nltk import word_tokenize, sent_tokenize, pos_tag
from nltk.data import load
from sklearn.naive_bayes import MultinomialNB
from collections import Counter
from time import time

start = time()

def load_list(resource_path):
    """load a newline separated text file of function words.
    Return a list"""
    vocab = []
    with open(resource_path, 'r') as f:
        for line in f:
            if line.strip():
                vocab.append(line.lower().strip())
    return vocab


def load_texts(path):
    authors = []
    texts = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            fields = line.lower().strip().split("\t")
            authors.append(fields[1])
            texts.append(fields[-1])
        return authors, texts, lines


def unison_shuffle(a, b, c):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    np.random.shuffle(c)


def split_dataset(X0, y0, hold_out_percent=0.9):
    """shuffle and split the dataset. Returns two tuples:
    (X_train, y_train, train_indices): train inputs
    (X_val, y_val, val_indices): validation inputs"""
    # index all data
    index = np.arange(X0.shape[0])
    X = np.copy(X0)
    y = np.copy(y0)
    unison_shuffle(X, y, index)
    split_point = int(np.ma.size(y) * hold_out_percent)
    X_train, X_val = X[:split_point,:], X[split_point:,:]
    y_train, y_val = y[:split_point], y[split_point:]
    train_indices, val_indices = index[:split_point], index[split_point:]
    #return ((X_train,y_train,np.arange(X.shape[0])), (X_val,y_val,np.arange(X.shape[0])))
    return ((X_train,y_train,train_indices), (X_val,y_val,val_indices))


def zero_rule_algorithm(train, test):
    """Gives a most common class baseline for the dataset as split into training and testing sets"""
    output_vals = [row for row in train]
    prediction = max(set(output_vals), key=output_vals.count)
    predicted = np.asarray([prediction for i in range(len(test))])
    return predicted


def accuracy_scorer(array1, array2):
    element_check = array1 == array2
    score = np.sum(element_check)
    return score


def narrow_scope(authors, lines):
    c = Counter(authors)
    top_2 = c.most_common(2)
    t2 = [i[0] for i in top_2]
    top_5 = c.most_common(5)
    t5 = [i[0] for i in top_5]
    top_10 = c.most_common(10)
    t10 = [i[0] for i in top_10]
    top_50 = c.most_common(50)
    t50 = [i[0] for i in top_50]

    narrow_2 = []
    narrow_5 = []
    narrow_10 = []
    narrow_50 =[]
    for line in lines:
        fields = line.lower().strip().split("\t")
        if fields[1] in t2:
            narrow_2.append(line)
            narrow_5.append(line)
            narrow_10.append(line)
            narrow_50.append(line)
        elif fields[1] in t5:
            narrow_5.append(line)
            narrow_10.append(line)
            narrow_50.append(line)
        elif fields[1] in t10:
            narrow_10.append(line)
            narrow_50.append(line)
        elif fields[1] in t50:
            narrow_50.append(line)
    return narrow_2, narrow_5, narrow_10, narrow_50


def narrow_thousands(authors, lines):
    unique_auths = list(set(authors))
    a2 = unique_auths[:2]
    a5 = unique_auths[:5]
    a10 = unique_auths[:10]
    a15 = unique_auths[:15]
    a20 = unique_auths[:20]
    a25 = unique_auths[:25]
    a30 = unique_auths[:30]
    a35 = unique_auths[:]

    n2 = []
    n5 = []
    #n10 = []
    n15 = []
    #n20 = []
    n25 = []
    #n30 = []
    n35 = []
    for line in lines:
        fields = line.lower().strip().split("\t")
        if fields[1] in a2:
            n2.append(line)
            n5.append(line)
            #n10.append(line)
            n15.append(line)
            #n20.append(line)
            n25.append(line)
            #n30.append(line)
            n35.append(line)

        elif fields[1] in a5:
            n5.append(line)
            #n10.append(line)
            n15.append(line)
            #n20.append(line)
            n25.append(line)
            #n30.append(line)
            n35.append(line)

        elif fields[1] in a10:
            #n10.append(line)
            n15.append(line)
            #n20.append(line)
            n25.append(line)
            #n30.append(line)
            n35.append(line)

        elif fields[1] in a15:
            n15.append(line)
            #n20.append(line)
            n25.append(line)
            #n30.append(line)
            n35.append(line)

        elif fields[1] in a20:
            #n20.append(line)
            n25.append(line)
            #n30.append(line)
            n35.append(line)

        elif fields[1] in a25:
            n25.append(line)
            #n30.append(line)
            n35.append(line)

        elif fields[1] in a30:
            #n30.append(line)
            n35.append(line)

        elif fields[1] in a35:
            n35.append(line)

    return n2, n5, n15, n25, n35


def nb_from_list(lines, vocab_path):
    """Build an authorship attribution classifier using MultinomialNaiveBayes for two authors.
    Returns a basic report of MultinomialNB predictions on the test data (10% of input),
    compared to a zero method baseline. Evaluates and reports the most accurate method."""

    list_vocab = load_list(vocab_path)

    authors = []
    texts = []
    for line in lines:
        fields = line.lower().strip().split("\t")
        authors.append(fields[1])
        texts.append(fields[-1])

    labels = np.asarray(authors)

    # Make review - f-word array
    review_features = np.zeros((len(texts), len(list_vocab)), dtype=np.int)
    for i, text in enumerate(texts):
        review_toks = word_tokenize(text)
        for j, function_word in enumerate(list_vocab):
            review_features[i, j] = len([w for w in review_toks if w == function_word])

    # split into training and testing arrays
    train, test = split_dataset(review_features, labels, 0.9)

    # create binary version of the same arrays
    train_binary = np.copy(train[0])
    test_binary = np.copy(test[0])

    def f(x):
        return 1 if x > 0 else 0
    f = np.vectorize(f)
    train_binary = f(train_binary)
    test_binary = f(test_binary)

    # implement and print score for zero rule classifier
    zero_rule_predictions = zero_rule_algorithm(train[1], test[0])

    # check accuracy
    zr_acc = accuracy_scorer(zero_rule_predictions, test[1])
    zr_pct = "%.3f" % (zr_acc / len(test[1]) * 100)

    # use MultinomialNB to predict classes
    nb_izer = MultinomialNB()
    nb_izer.fit(train[0], train[1])
    nb_predictions = nb_izer.predict(test[0])

    # check accuracy
    nb_acc = accuracy_scorer(nb_predictions, test[1])
    nb_pct = "%.3f" % (nb_acc / len(test[1]) * 100)
    print(f'{nb_acc} of {len(test[1])}, {nb_pct}% correct\n')

    return zr_pct, nb_pct


def sentences(line):
    sent_lengths = []
    sents = sent_tokenize(line)
    num_sents = len(sents)
    tok_lengths = []
    for sent in sents:
        toks = word_tokenize(sent)
        for tok in toks:
            tok_lengths.append(len(tok))
        sent_length = len(toks)
        sent_lengths.append(sent_length)
    mean_length = sum(sent_lengths)/len(sent_lengths)
    mean_tok = sum(tok_lengths)/len(tok_lengths)

    return num_sents, mean_length, mean_tok


def length_analysis(lines):
    # No number of sentences, just length of sentence

    features = ("num_sents", "mean_length", "mean_tok")
    authors = []
    texts = []
    for line in lines:
        fields = line.lower().strip().split("\t")
        authors.append(fields[1])
        texts.append(fields[-1])

    labels = np.asarray(authors)
    text_features = np.zeros((len(texts), len(features)), dtype=np.int)
    for i, text in enumerate(texts):
        stats = sentences(text)
        for j, stat in enumerate(stats):
            text_features[i, j] = stats[j]

    # split into training and testing arrays
    train, test = split_dataset(text_features, labels, 0.9)


    # implement and print score for zero rule classifier
    zero_rule_predictions = zero_rule_algorithm(train[1], test[0])

    # check accuracy
    zr_acc = accuracy_scorer(zero_rule_predictions, test[1])
    zr_pct = "%.3f" % (zr_acc / len(test[1]) * 100)

    # use MultinomialNB to predict classes
    nb_izer = MultinomialNB()
    nb_izer.fit(train[0], train[1])
    nb_predictions = nb_izer.predict(test[0])

    # check accuracy
    nb_acc = accuracy_scorer(nb_predictions, test[1])
    nb_pct = "%.3f" % (nb_acc / len(test[1]) * 100)

    return zr_pct, nb_pct


def pos_tagger(line):
    sents = sent_tokenize(line)
    line_tags = []
    for sent in sents:
        toks = word_tokenize(sent)
        tags = pos_tag(toks)
        just_tags = [tag[1] for tag in tags]
        line_tags += just_tags
    return line_tags


def pos_freqs(lines):
    tagdict = load('help/tagsets/upenn_tagset.pickle')
    keys = tagdict.keys()
    key_list = list(keys)
    authors = []
    texts = []
    for line in lines:
        fields = line.lower().strip().split("\t")
        authors.append(fields[1])
        texts.append(fields[-1])
    labels = np.asarray(authors)
    freqs_array = np.zeros((len(texts), len(key_list)), dtype=np.int)
    for i, text in enumerate(texts):
        tags = pos_tagger(text)
        for j, key in enumerate(key_list):
            freqs_array[i, j] = len([tag for tag in tags if tag == key])

    # split into training and testing arrays
    train, test = split_dataset(freqs_array, labels, 0.9)


    # implement and print score for zero rule classifier
    zero_rule_predictions = zero_rule_algorithm(train[1], test[0])

    # check accuracy
    zr_acc = accuracy_scorer(zero_rule_predictions, test[1])
    zr_pct = "%.3f" % (zr_acc / len(test[1]) * 100)

    # use MultinomialNB to predict classes
    nb_izer = MultinomialNB()
    nb_izer.fit(train[0], train[1])
    nb_predictions = nb_izer.predict(test[0])

    # check accuracy
    nb_acc = accuracy_scorer(nb_predictions, test[1])
    nb_pct = "%.3f" % (nb_acc / len(test[1]) * 100)

    return zr_pct, nb_pct


def bi_grams(lines, bigrams_list="top_100_bigrams.tsv"):
    # Counts instances of each bigrams in bigrams_list in each text
    # Returns a 2d numpy array of bigram counts

    bigram_tups = []
    with open(bigrams_list, 'r', encoding='utf-8') as h:
        bgs = h.readlines()
        for bg in bgs:
            ugs = bg.strip().split("\t")
            bigram_tup = (ugs[0], ugs[1])
            bigram_tups.append(bigram_tup)

    authors = []
    texts = []
    for line in lines:
        fields = line.lower().strip().split("\t")
        authors.append(fields[1])
        texts.append(fields[-1])

    labels = np.asarray(authors)

    bigram_freqs = np.zeros((len(texts), len(bigram_tups)), dtype=np.int)
    for i, doc in enumerate(texts):
        doc_bigrams = []
        doc_unigrams = doc.split(" ")
        for k in range(len(doc_unigrams) - 1):
            doc_bigrams.append((doc_unigrams[k], doc_unigrams[k + 1]))
        for j, bigram in enumerate(bigram_tups):
            count = len([b for b in doc_bigrams if b == bigram])
            bigram_freqs[i, j] = count

    # split into training and testing arrays
    train, test = split_dataset(bigram_freqs, labels, 0.9)

    # implement and print score for zero rule classifier
    zero_rule_predictions = zero_rule_algorithm(train[1], test[0])

    # check accuracy
    zr_acc = accuracy_scorer(zero_rule_predictions, test[1])
    zr_pct = "%.3f" % (zr_acc / len(test[1]) * 100)

    # use MultinomialNB to predict classes
    nb_izer = MultinomialNB()
    nb_izer.fit(train[0], train[1])
    nb_predictions = nb_izer.predict(test[0])

    # check accuracy
    nb_acc = accuracy_scorer(nb_predictions, test[1])
    nb_pct = "%.3f" % (nb_acc / len(test[1]) * 100)
    print(f'{nb_acc} of {len(test[1])}, {nb_pct}% correct\n')

    return zr_pct, nb_pct


def submain5(path, outfile="bigram_out.tsv"):
    authors, emails, all_texts = load_texts(path)
    top2, top5, top10, top50 = narrow_scope(authors, all_texts)
    corpus_slices = [top2, top5, top10, top50, all_texts]
    names = ["2", "5", "10", "50", "149"]

    for i in range(len(corpus_slices)):
        zr, nb = bi_grams(corpus_slices[i])
        with open(outfile, 'a', encoding='utf-8') as out:
            out.write(f"{names[i]}\t{zr}\t{nb}\n")


def submain10(path, outfile="bigram_k.tsv"):
    authors, emails, all_texts = load_texts(path)
    n2, n5, n15, n25, n35 = narrow_thousands(authors, all_texts)
    corpus_slices = [n2, n5, n15, n25, n35]
    names = ["2", "5", "15", "25", "35"]

    for i in range(len(corpus_slices)):
        zr, nb = bi_grams(corpus_slices[i])
        with open(outfile, 'a', encoding='utf-8') as out:
            out.write(f"{names[i]}\t{zr}\t{nb}\n")


def main(path1,path2):
    #   iterate through each submain function 10 times, acquire average results for plotting
    for _ in range(1):
        submain5(path1)
        submain10(path2)
        runtime = time() - start
        print(f"Runtime: {int(runtime // 60)} min {'%.2f' % (runtime % 60)} secs")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='feature vector homework')
    parser.add_argument('--path1', type=str, default="all_emails_dbl_clean.tsv",
                        help='path to author dataset')
    parser.add_argument('--path2', type=str, default="enron_thousands.tsv",
                        help='path to the leveled author dataset')
    # parser.add_argument('--punct_path', type=str, default="punct_list.txt",
    #                     help='path to the list of punctuation to use as features')
    args = parser.parse_args()

    main(args.path1, args.path2)
