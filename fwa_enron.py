"""
fwa_enron.py
- Tripp Maloney, 2019

This script takes a tsv extracted from the Enron email corpus and tests how effectively function word analysis can distinguish authorship.
"""

import numpy as np
import argparse
from sklearn.naive_bayes import MultinomialNB
from collections import Counter

def load_function_words(resource_path):
    """load a newline separated text file of function words.
    Return a list"""
    f_words = []
    with open(resource_path, 'r') as f:
        for line in f:
            if line.strip():
                f_words.append(line.lower().strip())
    return f_words


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


def split_dataset(X, y, hold_out_percent):
    """shuffle and split the dataset. Returns two tuples:
    (X_train, y_train, train_indices): train inputs
    (X_val, y_val, val_indices): validation inputs"""
    # index all data
    index = np.arange(X.shape[0])
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





def fwa(lines, vocab_path):
    """Build an authorship attribution classifier using MultinomialNaiveBayes for two authors.
    Returns a basic report of MultinomialNB predictions on the test data (10% of input),
    compared to a zero method baseline. Evaluates and reports the most accurate method."""

    function_words = load_function_words(vocab_path)

    reviews = []
    authors = []
    for line in lines:
            fields = line.strip().lower().split("\t")
            reviews.append(fields[-1])
            authors.append(fields[1])
    labels = np.asarray(authors)

    # Make review - f-word array
    review_features = np.zeros((len(reviews), len(function_words)), dtype=np.int)
    for i, review in enumerate(reviews):
        review_toks = review.split(" ")
        for j, function_word in enumerate(function_words):
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
    print("Baseline Predictions:")
    print(list(zero_rule_predictions))
    print()

    # check accuracy
    zr_acc = accuracy_scorer(zero_rule_predictions, test[1])
    zr_pct = "%.3f" % (zr_acc / len(test[1]) * 100)
    print(f'{zr_acc} of {len(test[1])}, {zr_pct}% correct\n')

    # use MultinomialNB to predict classes
    nb_izer = MultinomialNB()
    nb_izer.fit(train[0], train[1])
    nb_predictions = nb_izer.predict(test[0])
    print("Naive Bayes Predictions:")
    print(list(nb_predictions))
    print()

    # check accuracy
    nb_acc = accuracy_scorer(nb_predictions, test[1])
    nb_pct = "%.3f" % (nb_acc / len(test[1]) * 100)
    print(f'{nb_acc} of {len(test[1])}, {nb_pct}% correct\n')

    # re-implement on binary array
    nb_izer.fit(train_binary, train[1])
    nb_binary = nb_izer.predict(test_binary)
    print("Binary NB Predictions:")
    print(list(nb_binary))
    print()

    # check accuracy
    bin_acc = accuracy_scorer(nb_binary, test[1])
    bin_pct = "%.3f" % (bin_acc / len(test[1]) * 100)
    print(f'{bin_acc} of {len(test[1])}, {bin_pct}% correct\n')

    # print the actual authors for comparison. This is probably superfluous.
    print("Actual Authors:")
    print(list(test[1]))
    print()

    # determine best test on this run
    if max(zr_acc, nb_acc, bin_acc) == zr_acc:
        print("Tests did not beat baseline.")
    elif nb_acc == bin_acc > zr_acc:
        print("Best test: Naive Bayes, integers OR boolean.")
    elif max(zr_acc, nb_acc, bin_acc) == nb_acc:
        print("Best method: Naive Bayes with integers.")
    elif max(zr_acc, nb_acc, bin_acc) == bin_acc:
        print("Best method: Naive Bayes with booleans.")
    else:
        print("Something strange has happened.")

    return zr_pct, nb_pct, bin_pct


def main(path, fwlist, outfile='fwa_results.txt'):
    authors, emails, all_texts = load_texts(path)
    top2, top5, top10, top50 = narrow_scope(authors, all_texts)
    corpus_slices = [top2, top5, top10, top50, all_texts]
    names = ["2", "5", "10", "50", "149"]

    for i in range(len(corpus_slices)):
        zr, nb, bin = fwa(corpus_slices[i], fwlist)
        with open(outfile, 'a', encoding='utf-8') as out:
            out.write(f"{names[i]}\t{zr}\t{nb}\t{bin}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='feature vector homework')
    parser.add_argument('--path', type=str, default="all_emails_dbl_clean.tsv",
                        help='path to author dataset')
    parser.add_argument('--function_words_path', type=str, default="ewl_function_words.txt",
                        help='path to the list of words to use as features')
    args = parser.parse_args()

    main(args.path, args.function_words_path)
