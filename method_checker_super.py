"""
method_evaluator.py
By Tripp Maloney
Last edited 20190423

This script is meant to evaluate different computational authorship attribution methods on the Enron email dataset.
The intended input for this script is a comprehensive .tsv containing of the following format:

email#  authname    email-text
001537	arnold-j	how was the concert?
"""

from collections import Counter
import numpy as np
import argparse
from nltk import word_tokenize, sent_tokenize, pos_tag
from nltk.data import load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
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


def unison_shuffle(arr_list_in, labels_in):
    arr_list = np.copy(arr_list_in)
    labels = np.copy(labels_in)
    # input: list of numpy arrays to shuffle
    rng_state = np.random.get_state()
    np.random.shuffle(arr_list)
    np.random.set_state(rng_state)
    np.random.shuffle(labels)
    return arr_list, labels


def split_dataset(arr_in, labels_in, hold_out_percent=0.9):
    """shuffle and split the dataset. Returns two tuples:
    (X_train, y_train, train_indices): train inputs
    (X_val, y_val, val_indices): validation inputs"""
    # index all data
    X, y = unison_shuffle(arr_in, labels_in)
    split_point = int(np.ma.size(y) * hold_out_percent)

    y_train, y_val = y[:split_point], y[split_point:]
    X_train, X_val = X[:split_point, :], X[split_point:, :]

    #return ((X_train,y_train,np.arange(X.shape[0])), (X_val,y_val,np.arange(X.shape[0])))
    return (X_train, y_train), (X_val, y_val)


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


def narrow_arrays(array, labels):
    c = Counter(labels)
    top_2 = c.most_common(2)
    t2 = [i[0] for i in top_2]
    top_2_indices = [i for i in range(len(labels)) if labels[i] in t2]
    t2arr = array[top_2_indices]

    top_5 = c.most_common(5)
    t5 = [i[0] for i in top_5]
    top_5_indices = [i for i in range(len(labels)) if labels[i] in t5]
    t5arr = array[top_5_indices]

    top_10 = c.most_common(10)
    t10 = [i[0] for i in top_10]
    top_10_indices = [i for i in range(len(labels)) if labels[i] in t10]
    t10arr = array[top_10_indices]

    top_50 = c.most_common(50)
    t50 = [i[0] for i in top_50]
    top_50_indices = [i for i in range(len(labels)) if labels[i] in t50]
    t50arr = array[top_50_indices]

    return t2arr, t5arr, t10arr, t50arr



def features_from_list(texts, vocab_path):
    """Build an authorship attribution classifier using MultinomialNaiveBayes for two authors.
    Returns a basic report of MultinomialNB predictions on the test data (10% of input),
    compared to a zero method baseline. Evaluates and reports the most accurate method."""

    list_vocab = load_list(vocab_path)

    # Make review - f-word array
    review_features = np.zeros((len(texts), len(list_vocab)), dtype=np.int)
    for i, text in enumerate(texts):
        review_toks = word_tokenize(text)
        for j, function_word in enumerate(list_vocab):
            review_features[i, j] = len([w for w in review_toks if w == function_word])

    return review_features


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


def length_analysis(texts):
    # No number of sentences, just length of sentence

    features = ("num_sents", "mean_length", "mean_tok")

    text_features = np.zeros((len(texts), len(features)), dtype=np.int)
    for i, text in enumerate(texts):
        stats = sentences(text)
        for j, stat in enumerate(stats):
            text_features[i, j] = stats[j]

    return text_features


def pos_tagger(line):
    sents = sent_tokenize(line)
    line_tags = []
    for sent in sents:
        toks = word_tokenize(sent)
        tags = pos_tag(toks)
        just_tags = [tag[1] for tag in tags]
        line_tags += just_tags
    return line_tags


def pos_freqs(texts):

    tagdict = load('help/tagsets/upenn_tagset.pickle')
    keys = tagdict.keys()
    key_list = list(keys)

    freqs_array = np.zeros((len(texts), len(key_list)), dtype=np.int)
    for i, text in enumerate(texts):
        tags = pos_tagger(text)
        for j, key in enumerate(key_list):
            freqs_array[i, j] = len([tag for tag in tags if tag == key])

    return freqs_array


def bi_grams(texts, bigrams_list="top_bigrams.tsv"):
    # Counts instances of each bigrams in bigrams_list in each text
    # Returns a 2d numpy array of bigram counts

    bigram_tups = []
    with open(bigrams_list, 'r', encoding='windows-1256') as h:
        lines = h.readlines()
        for line in lines:
            fields = line.strip().split("\t")
            bigram_tup = (fields[0], fields[1])
            bigram_tups.append(bigram_tup)

    bigram_freqs = np.zeros((len(texts), len(bigram_tups)), dtype=np.int)
    for i, text in enumerate(texts):
        doc_bigrams = []
        doc_unigrams = text.split(" ")
        for k in range(len(doc_unigrams) - 1):
            doc_bigrams.append((doc_unigrams[k], doc_unigrams[k + 1]))
        for j, bigram in enumerate(bigram_tups):
            count = len([bg for bg in doc_bigrams if bg == bigram])
            bigram_freqs[i, j] = count

    return bigram_freqs


def tf_idf(texts):
    # Runs TF-IDF measurements on a given set of texts
    # Returns a 2d array of TF-IDF measures

    vectorizer = TfidfVectorizer("content", lowercase=True, analyzer="word", use_idf=True, min_df=10)
    X = vectorizer.fit_transform(texts).toarray()

    return X


def nb_train_test(feature_arr, labels):

    # split into training and testing arrays
    train, test = split_dataset(feature_arr, labels, 0.9)

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
    nb_pct.append(nb_pct)

    return zr_pct, nb_pct


def method_compiler(texts):

    fwa = features_from_list(texts, 'ewl_function_words.txt')
    punct = features_from_list(texts, 'punct_list.txt')
    length = length_analysis(texts)
    pos = pos_freqs(texts)
    bg = bi_grams(texts)
    # tf = tf_idf(texts)

    return [fwa, punct, length, pos, bg]


def all_auths_eval(path, outfile='all_methods_out.tsv'):
    authors, texts, lines = load_texts(path)
    labels = np.asarray(authors)
    names = ["2", "5", "10", "50", "149"]

    feature_arrs = method_compiler(texts)
    for arr in feature_arrs:
        narrow_arrs = narrow_arrays(arr, labels)
        for n_arr in narrow_arrs:
            zr, nb = nb_train_test(n_arr, labels)
            with open(outfile, 'a', encoding='utf-8') as out:
                out.write(f"{zr}\t{nb}\t")
        zr, nb = nb_train_test(arr, labels)
        with open(outfile, 'a', encoding='utf-8') as out:
            out.write(f"{zr}\t{nb}\t\n")



def narrow_thousands(tsv):
    authors, texts, lines = load_texts(tsv)

    unique_auths = set(list(authors))
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
    n10 = []
    n15 = []
    n20 = []
    n25 = []
    n30 = []
    n35 = []
    for line in lines:
        fields = line.lower().strip().split("\t")
        if fields[1] in a2:
            n2.append(line)
            n5.append(line)
            n10.append(line)
            n15.append(line)
            n20.append(line)
            n25.append(line)
            n30.append(line)
            n35.append(line)

        elif fields[1] in a5:
            n5.append(line)
            n10.append(line)
            n15.append(line)
            n20.append(line)
            n25.append(line)
            n30.append(line)
            n35.append(line)

        elif fields[1] in a10:
            n10.append(line)
            n15.append(line)
            n20.append(line)
            n25.append(line)
            n30.append(line)
            n35.append(line)

        elif fields[1] in a15:
            n15.append(line)
            n20.append(line)
            n25.append(line)
            n30.append(line)
            n35.append(line)

        elif fields[1] in a20:
            n20.append(line)
            n25.append(line)
            n30.append(line)
            n35.append(line)

        elif fields[1] in a25:
            n25.append(line)
            n30.append(line)
            n35.append(line)

        elif fields[1] in a30:
            n30.append(line)
            n35.append(line)

        elif fields[1] in a35:
            n35.append(line)

    return n2, n5, n10, n15, n20, n25, n30, n35


def thousands_eval(path, outfile = 'thousands_out.tsv'):
    authors, texts, lines = load_texts(path)
    n2, n5, n10, n15, n20, n25, n30, n35 = narrow_scope(authors, lines)
    corpus_slices = [n2, n5, n10, n15, n20, n25, n30, n35]
    names = ["2", "5", "10", "15", "20", "25", "30", "35"]

    for i in range(len(corpus_slices)):
        authors = []
        for line in corpus_slices[i]:
            fields = line.strip().split("\t")
            authors.append(fields[1])
            texts.append(fields[-1])
        labels = np.asarray(authors)
        with open(outfile, 'a', encoding='utf-8') as out:
            out.write(f"{names[i]}\t")
        feature_arrs = method_compiler(texts)
        for arr in feature_arrs:
            zr, nb = nb_train_test(arr, labels)
            with open(outfile, 'a', encoding='utf-8') as out:
                out.write(f"{zr}\t{nb}\t")
        with open(outfile, 'a', encoding='utf-8') as out:
            out.write("\n")


def main(path1, path2):
    #   iterate through each submain function 10 times, acquire average results for plotting
    for _ in range(1):
        all_auths_eval(path1)
        #thousands_eval(path2)
        runtime = time() - start
        print(f"Runtime: {int(runtime // 60)} min {'%.2f' % (runtime % 60)} secs")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='feature vector homework')
    parser.add_argument('--path1', type=str, default="all_emails_dbl_clean.tsv",
                        help='path to author dataset')
    parser.add_argument('--path2', type=str, default="enron_thousands.tsv",
                        help='path to limited author dataset')
    args = parser.parse_args()

    main(args.path1, args.path2)
