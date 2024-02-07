# ====================== #
# Anmod 1: N-gram models #
# ====================== #

import itertools
import math
import random
import nltk

# -----
# Preparation
# -----

# Imports a txt file (containing one sentence per line, w/o sentence-ending
# periods) as a list of lists of words:
def txt_import(filename):
    """Import a txt list of sentences as a list of lists of words.
    
    Argument:
        - filename (string), e.g.: 'grimm_corpus.txt'
    
    Returns:
        - list (of lists of strings), e.g.:
          [['my', 'name', 'is', 'jolán'], ['i', 'am', 'cool'], ..., ['bye']]
    """
    with open(filename, mode='r', encoding='utf-8-sig') as file:
        lines = file.readlines()
    return [line.strip().split() for line in lines]

# Randomly sorts a list of sentences into 90% training and 10% testing data:
def train_test(corpus):
    """Randomly sort corpus into 90pct training and 10pct testing data.
    
    Testing data is filtered to only contain sentences whose words all occur in
    the training data, so actual testing data may be smaller than 10 percent.
    
    Argument:
        - corpus (list of lists of words /> output of txt_import()), e.g.:
          [['my', 'name', 'is', 'jolán'], ['i', 'am', 'cool'], ..., ['bye']]
    
    Returns:
        - (train, test) where train is randomly selected 90pct of corpus, and
          test is remaining 10pct filtered s.t. none of its sentences contain
          words that aren't in train /> train: input of *_model()
    """
    sentences = corpus.copy()
    random.shuffle(sentences)
    n = round(len(sentences) * 0.9)
    train = sentences[:n]
    vocab = {word for sentence in train for word in sentence}
    test = [sentence for sentence in sentences[n:]
                     if set(sentence).issubset(vocab)]
    return (train, test)

# Computes word frequency dictionary from corpus, where keys are nltk's pos
# tags and values are lists of (word, freq) pairs, sorted biggest-first by freq:
def freq_dict(corpus):
    """Sort words from a list of sentences by frequency, per word class.
    
    Argument:
        - corpus (list of lists of strings), e.g.:
          [['a', 'certain', 'king', 'had', 'a', 'beautiful', 'garden'],
           ['in', 'the', 'garden', 'stood', 'a', 'tree'], ...]
    
    Returns:
        - dict (of {word class: [(word, freq), ...]} items), e.g.:
          {'DT': [('the', 6770), ('a', 1909), ('all', 412), ...],
           'JJ': [('little', 392), ('good', 203), ('old', 200), ...], ...}
    """
    # Compute unsorted frequency dicts per word class:
    wc_dy = {}
    for sentence in corpus:
        for word, wc in nltk.pos_tag(sentence):
            try:
                value_or_zero = wc_dy[wc].setdefault(word, 0)
                wc_dy[wc].update({word: value_or_zero + 1})
            except KeyError:
                wc_dy.update({wc: {word: 1}})
    # Sort the (word, freq) pairs by freq from big to small in each word class:
    freq_dy = {}
    for wc in wc_dy:
        sorted_word_freqs = sorted(wc_dy[wc].items(),
                                   key=lambda x: x[1],
                                   reverse=True)
        freq_dy[wc] = sorted_word_freqs
    return freq_dy

# -----
# Auxiliary functions for model building
# -----

# Computes list of n-grams of sentence:
def ngrams_list(sentence, n):
    """Compute list of n-grams (n-tuples of words) of sentence (list of words).
    
    (Attaches appropriate number of sentence-markers at beginning and end.)
    
    Arguments:
        - sentence (list of strings), e.g.: ['i', 'am', 'cool']
        - n (integer), e.g.: 2
    
    Returns:
        - list (of tuples of strings), e.g.:
          [('<s>', 'i'), ('i', 'am'), ('am', 'cool'), ('cool', '</s>')]
    """
    sentence = (n-1) * ['<s>',] + sentence + ['</s>',]
    return list(zip(*(sentence[i:len(sentence)-n+i+1] for i in range(n))))

def ctxt_dict_upto(corpus, n):
    cdy = {}
    for sentence in corpus:
        for ngram in ngrams_list(sentence, n):
            goal = ngram[-1:]
            for i in range(len(ngram)-1):
                ctxt = ngram[i:-1]
                try:
                    value_or_zero = cdy[ctxt].setdefault(goal, 0)
                    cdy[ctxt].update({goal: value_or_zero + 1})
                except KeyError:
                    cdy.update({ctxt: {goal: 1}})
    return cdy

def ctxt_dict(corpus, n):
    cdy = {}
    for sentence in corpus:
        for ngram in ngrams_list(sentence, n):
            ctxt, goal = ngram[:-1], ngram[-1:]
            try:
                value_or_zero = cdy[ctxt].setdefault(goal, 0)
                cdy[ctxt].update({goal: value_or_zero + 1})
            except KeyError:
                cdy.update({ctxt: {goal: 1}})
    return cdy

# -----
# Model building
# -----

# Maximum likelihood estimate
def mle_model(corpus, n=2):
    cdy = ctxt_dict(corpus, n)
    pdy = {}
    for ctxt in cdy:
        ctxt_freq = sum(cdy[ctxt].values())
        for goal in cdy[ctxt]:
            ctxt_goal_freq = cdy[ctxt][goal]
            pdy[ctxt + goal] = ctxt_goal_freq / ctxt_freq
    return {'pdy': pdy, 'order': n}

def mle_prob(sentence, model):
    pdy, n = model['pdy'], model['order']
    ngrams = ngrams_list(sentence, n)
    prob = 1
    for ngram in ngrams:
        try:
            prob *= pdy[ngram]
        except KeyError:
            return 0
    return prob

# Lidstone ("add-k") smoothing
def lid_model(corpus, n=2, k=0.05):
    cdy = ctxt_dict(corpus, n)
    vocab_size = len(set(itertools.chain(*(cdy[ctxt].keys() for ctxt in cdy))))
    pdy = {}
    ctxt_freqs = {}
    for ctxt in cdy:
        ctxt_freq = sum(cdy[ctxt].values()) + (k * vocab_size)
        ctxt_freqs[ctxt] = ctxt_freq
        for goal in cdy[ctxt]:
            ngram_freq = cdy[ctxt][goal] + k
            pdy[ctxt + goal] = ngram_freq / ctxt_freq
    return {'pdy': pdy, 'order': n, 'add_param': k,
            'ctxt_freqs': ctxt_freqs, 'vocab_size': vocab_size}

def lid_prob(sentence, model):
    pdy, n, k = model['pdy'], model['order'], model['add_param']
    ctxt_freqs, vocab_size = model['ctxt_freqs'], model['vocab_size']
    ngrams = ngrams_list(sentence, n)
    prob = 1
    for ngram in ngrams:
        try:
            prob *= pdy[ngram]
        except KeyError:
            ctxt_freq = ctxt_freqs.setdefault(ngram[:-1], k * vocab_size)
            prob *= k / ctxt_freq
    return prob

# Baseline interpolation
def bip_model(corpus, n=2, lambdas=(0.75, 0.25)):
    cdy = ctxt_dict_upto(corpus, n)
    pdy = {}
    

# -----
# Should we have multiple sentence-end markers? (Tentatively: no.)
# -----

# Generates list of all letter sequences (as lists) up to n over letters:
def strings_upto(letters, n, strings=[], prev_strings=[[]]):
    if n == 0:
        return strings
    else:
        new_strings = [[letter] + prev_string for letter in letters
                                              for prev_string in prev_strings]
        return strings_upto(letters, n-1, strings + new_strings, new_strings)