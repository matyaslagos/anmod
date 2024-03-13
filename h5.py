from collections import defaultdict
#from math import log2, exp2
from random import shuffle

# -----
# Preparing the observation and estimation data
# -----

# Imports a txt file (containing one sentence per line, w/o sentence-ending
# periods) as a list of lists of words
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

# Randomly sorts a corpus into 50% observation data and 50pct estimation data.
def obs_est(corpus):
    """Randomly sort corpus into 50pct observ. data and 50pct estimation data.
    
    Arguments:
        - corpus (list of lists of words), e.g.:
          [['my', 'name', 'is', 'jolán'], ['i', 'am', 'cool'], ..., ['bye']]
    
    Returns:
        - (obs, est) where obs and est both contain random half of corpus.
    """
    sentences = corpus.copy()
    shuffle(sentences)
    n = round(len(sentences) * 0.5)
    obs = sentences[:n]
    est = sentences[n:]
    return (obs, est)

# -----
# Auxiliary functions
# -----

# Gets list of bigrams (2-tuples of strings) of sentence (list of strings)
def bigrams_list(sentence):
    """Compute list of bigrams (2-tuples of words) of sentence (list of words).
    
    (Attaches sentence-markers at beginning and end.)
    
    Argument:
        - sentence (list of strings), e.g.: ['i', 'am', 'reading']
    
    Returns:
        - list (of tuples of strings), e.g.:
          [('<s>', 'i'), ('i', 'am'), ('am', 'reading'), ('reading', '</s>')]
    """
    return list(zip(['<s>'] + sentence, sentence + ['</s>']))

# Records forw. ('fw') and backw. ('bw') cooccurrence frequencies into a dict
def frequencies(training_data):
    """Record forward ('fw') and backward ('bw') cooccurrence freqs into a dict.
    
    Argument:
        - training_data (list of lists of strings), e.g.:
          [['my', 'name', 'is', 'jolán'], ['i', 'am', 'cool'], ..., ['bye']]
    
    Returns:
        - freq_dict (dict with two subdicts 'fw' and 'bw'), e.g.:
          > freq_dict['fw']['egy']['vízesést'] == 4
            - means: 'vízesést' occurred 4 times after 'egy', and
          > freq_dict['bw']['reggelt']['jó'] == 10
            - means: 'jó' occurred 10 times before 'reggelt'
    """
    freq_dict = {'fw': defaultdict(lambda: defaultdict(int)),
                 'bw': defaultdict(lambda: defaultdict(int))}
    for sentence in training_data:
        for bigram in bigrams_list(sentence):
            # If e.g. ('egy', 'vízesést') is an attested bigram, then...
            # (a) record that 'vízesést' occurred after 'egy'
            freq_dict['fw'][bigram[0]][bigram[1]] += 1
            # (b) record that 'egy' occurred before 'vízesést'
            freq_dict['bw'][bigram[1]][bigram[0]] += 1
    return freq_dict