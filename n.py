# ====================== #
# Anmod 1: N-gram models #
# ====================== #

from collections import defaultdict
from itertools import chain
from math import log2, exp2
from random import shuffle, choices

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
def train_test(corpus, no_unknowns=True):
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
    shuffle(sentences)
    n = round(len(sentences) * 0.9)
    train = sentences[:n]
    test = sentences[n:]
    if no_unknowns:
        vocab = {word for sentence in train for word in sentence}
        test = [sentence for sentence in test if set(sentence).issubset(vocab)]
    return (train, test)

# Creates word frequency dictionary from corpus, where keys are nltk's pos
# tags and values are lists of (word, freq) pairs, sorted biggest-first by freq
# (wrtten with Réka Bandi)
def freq_dict(corpus):
    """For each word class, get list of its most-to-least-frequent words.
    
    Argument:
        - corpus (list of lists of strings), e.g.:
          [['a', 'certain', 'king', 'had', 'a', 'beautiful', 'garden'],
           ['in', 'the', 'garden', 'stood', 'a', 'tree'], ...]
    
    Returns:
        - dict (of form {word class: [(word, freq), ...]}), e.g.:
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

# Collects n-grams of sentence into a list:
def ngrams_list(sentence, n):
    """Compute list of n-grams (n-tuples of words) of sentence (list of words).
    
    (Attaches appropriate number of sentence-markers at beginning and end.)
    
    Arguments:
        - sentence (list of strings), e.g.: ['i', 'am', 'reading']
        - n (integer), e.g.: 2
    
    Returns:
        - list (of tuples of strings), e.g.:
          [('<s>', 'i'), ('i', 'am'), ('am', 'reading'), ('reading', '</s>')]
    """
    sentence = ['<s>'] * (n-1) + sentence + ['</s>']
    return list(zip(*(sentence[i:len(sentence)-n+i+1] for i in range(n))))

def ctxt_dict_upto(corpus, n):
    cdy = {}
    for sentence in corpus:
        for ngram in ngrams_list(sentence, n):
            goal = ngram[-1:]
            for i in range(len(ngram)):
                ctxt = ngram[i:-1]
                try:
                    value_or_zero = cdy[ctxt].setdefault(goal, 0)
                    cdy[ctxt].update({goal: value_or_zero + 1})
                except KeyError:
                    cdy.update({ctxt: {goal: 1}})
    return cdy

# Creates ngram context dictionary for corpus
def ctxt_dict(corpus, n):
    cdy = {}
    for sentence in corpus:
        for ngram in ngrams_list(sentence, n):
            ctxt, goal = ngram[:-1], ngram[-1:]
            try:
                # If ctxt is in cdy, add one to its freq. with goal
                value_or_zero = cdy[ctxt].setdefault(goal, 0)
                cdy[ctxt].update({goal: value_or_zero + 1})
            except KeyError:
                # Else add ctxt to cdy and record this occurrence with goal
                cdy.update({ctxt: {goal: 1}})
    return cdy

# -----
# Model building
# -----

# Maximum likelihood estimate
def mle_model(corpus, n):
    cdy = ctxt_dict(corpus, n)
    pdy = {}
    for ctxt in cdy:
        ctxt_freq = sum(cdy[ctxt].values())
        for goal in cdy[ctxt]:
            ngram_freq = cdy[ctxt][goal]
            pdy[ctxt + goal] = ngram_freq / ctxt_freq
    return {'pdy': pdy, 'order': n, 'cdy': cdy}

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
def lid_model(corpus, n, k):
    cdy = ctxt_dict(corpus, n)
    # Get vocabulary size for later calculating total added frequency to ctxts
    vocab_size = len(set(chain(*(cdy[ctxt].keys() for ctxt in cdy))))
    pdy = {}
    ctxt_freqs = {}
    for ctxt in cdy:
        # Add total added freq. to ctxt
        ctxt_freq = sum(cdy[ctxt].values()) + (k * vocab_size)
        ctxt_freqs[ctxt] = ctxt_freq
        for goal in cdy[ctxt]:
            # Add appropriate freq. to ngram and calculate new ngram probability
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

# Baseline interpolation (weights are highest-order-first)
def itp_model(corpus, n, weights):
    cdy = ctxt_dict_upto(corpus, n)
    pdy = {}
    for ctxt in cdy:
        ctxt_freq = sum(cdy[ctxt].values())
        for goal in cdy[ctxt]:
            ngram_freq = cdy[ctxt][goal]
            pdy[ctxt + goal] = ngram_freq / ctxt_freq
    return {'pdy': pdy, 'order': n, 'weights': weights, 'cdy': cdy}

def itp_prob(sentence, model):
    pdy, n, weights = model['pdy'], model['order'], model['weights']
    ngrams = ngrams_list(sentence, n)
    prob = 1
    for ngram in ngrams:
        prob *= sum(weight * pdy.setdefault(ngram[i:], 0)
                    for i, weight in enumerate(weights))
    return prob

def itp_seq_prob(sequence, model):
    pdy, n, weights = model['pdy'], model['order'], model['weights']
    prob = sum(weight * pdy.setdefault(tuple(sequence[i:]), 0)
               for i, weight in enumerate(weights))
    return prob

# Interpolated Kneser-Ney smoothing
def kn_model(corpus, n):
    train, held_out = train_test(corpus, no_unknowns=False)
    cdy = ctxt_dict_upto(train, n)
    disc_dy = {i : 0 for i in range(n)}
    for ctxt in cdy:
        pass

# Analogical paths
def ap2_model(train):
    # TODO: add comments to lines
    """Return bigram probability dictionary from context dictionary.
    
    Computes the "probability dictionary" of each word in `cdy` ("context
    dictionary", output of `ctxt_dy()`), containing the joint, backward
    transitional, and forward transitional probabilities of each bigram based
    on the co-occurrence data in `cdy`.
    
    > E.g. if `pdy` is a probability dictionary, then `pdy[('the', 'king')]` is
    `{'joint': 0.1, 'left': 0.6, 'right': 0.002}` if the joint probability of
    `('the', 'king')` is  0.1, its backward transitional probability is 0.6,
    and its forward transitional probability is 0.002.
    
    Keyword arguments:
    cdy -- "context dictionary" containing co-occurrence data for words,
        output of `ctxt_dy()`
    
    Returns:
    dict -- `pdy`, a "probability dictionary" containing the joint, backward
        transitional, and forward transitional probability of each bigram based
        on the co-occurrence data in `cdy`
    """
    cdy = ctxt_dy2(train)
    # TODO: record total_freq with ctxt_dy() so that no need to count again
    total_freq = sum(sum(cdy[key]['right'].values()) for key in cdy)
    pdy = {}
    for w1 in cdy:
        pdy[('_', w1)] = sum(cdy[w1]['left'].values()) / total_freq
        pdy[(w1, '_')] = sum(cdy[w1]['right'].values()) / total_freq
        for w2 in cdy[w1]['right']:
            count = cdy[w1]['right'][w2]
            joint = count / total_freq
            left = count / sum(cdy[w2]['left'].values())
            right = count / sum(cdy[w1]['right'].values())
            pdy[(w1, w2)] = {'joint': joint, 'left': left, 'right': right}
    # Compute anl. path weights through each attested bigram (a,b):
    anl_probs = {}
    for (a,b) in pdy:
        if '_' in (a,b):
            anl_probs[(a,b)] = pdy[(a,b)]
        else:
            for s in cdy[b]['left']:
                for t in cdy[a]['right']:
                    try:
                        anl_probs[(s,t)] += pdy[(a,b)]['joint']   \
                                            * pdy[(s,b)]['left']  \
                                            * pdy[(a,t)]['right']
                    except:
                        anl_probs[(s,t)]  = pdy[(a,b)]['joint']   \
                                            * pdy[(s,b)]['left']  \
                                            * pdy[(a,t)]['right']
    return (anl_probs, cdy, pdy)

def ctxt_dy2(train):
    # TODO: add comments for lines
    """Return bigram context dictionary from training data.
    
    Computes the "context dictionary" of each word in `train`. This dictionary
    maps each word to its left and right context dictionaries: the left context
    dictionary consists of pairs (left neighbor, freq), where `left neighbor`
    is a word that occurs in `train` directly before the word, and `freq` is
    the number of times these two words co-occur in this order; the right
    context dictionary is the same but for right neighbors.
    > E.g. if `cdy` is a context dictionary, then cdy['king']['left']['the']
    is 14 if the bigram ('the', 'king') occurs 14 times in the sentences
    of `train`.
    
    Keyword arguments:
    train -- list of lists of words (output of `txt_to_list()` or of
        `list_to_train_test()[0]`)
    
    Returns:
    dict -- `cdy`, where cdy['king']['left']['the'] is 14 if the bigram
        ('the', 'king') occurs 14 times in the sentences of `train`, and
        cdy['king']['right']['was'] is 18 if the bigram ('king', 'was') occurs
        18 times in the sentences of `train`
    """
    vocab = {word for sentence in train for word in sentence}
    vocab = vocab.union({'<s>', '</s>'})
    cdy = {word: {'left':{}, 'right':{}} for word in vocab}
    for sentence in train:
        padded = ['<s>'] + sentence + ['</s>']
        bigrams = zip(padded[:-1], padded[1:])
        for first, second in bigrams:
            try:
                cdy[first]['right'][second] += 1
            except KeyError:
                cdy[first]['right'][second] = 1
            try:
                cdy[second]['left'][first] += 1
            except KeyError:
                cdy[second]['left'][first] = 1
    return cdy

def ap2_prob(sentence, model):
    """Return the analogical log-probability of sentence according to model.
    
    Computes the analogical log-probability of sentence by adding together
    the analogical log-probabilities of the bigrams that occur in it. The
    probabilities of the bigrams (s,t) interpolate the following:
    - mle transitional probability of (s,t),
    - analogical-path transitional probability of (s,t), and
    - mle unigram probability of (_,t).
    
    Keyword arguments:
    sentence -- a list of strings
    model    -- (anl_probs, cdy, pdy), output of ap2_model()
    
    Returns:
    float -- logarithm of the probability of sentence according to model
    """
    bigrams = zip(['<s>'] + sentence, sentence + ['</s>'])
    anl_probs, cdy, pdy = model
    p = 1
    for s, t in bigrams:
        # (1) Get mle transitional probability of (s,t):
        try:
            mle_st = pdy[(s,t)]['right']
        except:
            mle_st = 0
        # (2a) Get anl-path probability of (s,t):
        try:
            ap_st = anl_probs[(s,t)]
        except:
            ap_st = 0
        # (2b) Get anl-path probability of (s,_) (same as its mle probability):
        ap_s = pdy[(s,'_')]
        # (2c) Get anl-path transitional probability of (s,t):
        anl_st = ap_st / ap_s
        # (3) Get mle probability of (_,t):
        mle_t = pdy[('_',t)]
        # Interpolate (1) mle_st, (2) anl_st, and (3) mle_t:
        intp_st = (0.595 * mle_st) + (0.4 * anl_st) + (0.005 * mle_t)
        p *= intp_st
    return p

# -----
# Generating sentences
# -----

# Generates word sequence of length at most 30 according to model probabilities
def mle_gen(model):
    cdy, order = model['cdy'], model['order']
    context = tuple('<s>' for _ in range(order-1))
    sentence = ''
    for _ in range(30):
        next_word = choices(list(cdy[context]), cdy[context].values())[0][0]
        if next_word == '</s>':
            break
        else:
            sentence += next_word + ' '
            context = (context + (next_word,))[1:]
    return sentence[:-1]

def itp_gen(model):
    cdy, order = model['cdy'], model['order']
    context = ['<s>' for _ in range(order-1)]
    sentence = ''
    vbly = [word[0] for word in cdy[()]]
    for _ in range(30):
        probs = [itp_seq_prob(context + [word], model) for word in vbly]
        next_word = choices(vbly, probs)[0]
        if next_word == '</s>':
            break
        else:
            sentence += next_word + ' '
            context = (context + [next_word])[1:]
    return sentence[:-1]

# -----
# Testing
# -----

# Calculates perplexity of model on test set
def perplexity(test, model, prob_function):
    rate = 1 / sum(len(sentence) + 1 for sentence in test)
    cross_entropy = sum(log2(1 / prob_function(sentence, model))
                        for sentence in test)
    return exp2(cross_entropy * rate)

# -----
# Should we have multiple sentence-end markers? (Tentatively: no.)
# -----

# Generates list of all letter sequences (as lists) up to n over letters
# (for testing whether sequence probabilities sum to 1)
def strings_upto(letters, n, strings=[], prev_strings=[[]]):
    if n == 0:
        return strings
    else:
        new_strings = [[letter] + prev_string for letter in letters
                                              for prev_string in prev_strings]
        return strings_upto(letters, n-1, strings + new_strings, new_strings)

# -----
# Homework
# -----

