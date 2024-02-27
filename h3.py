from collections import defaultdict
from math import log2, exp2
from random import shuffle

# -----
# Preparing the training and test data
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

# Randomly sorts a list of sentences into 90% training and 10% testing data
def train_test(corpus, no_unknowns=True):
    """Randomly sort corpus into 90pct training and 10pct testing data.
    
    (Testing data is by default filtered to only contain sentences whose words
    all occur in the training data, so actual testing data may be smaller than
    10 percent.)
    
    Arguments:
        - corpus (list of lists of words), e.g.:
          [['my', 'name', 'is', 'jolán'], ['i', 'am', 'cool'], ..., ['bye']]
        - no_unknowns (boolean, True by default): option for filtering out
          unseen words from the test data
    
    Returns:
        - (train, test) where train is randomly selected 90pct of corpus, and
          test is remaining 10pct, by default filtered s.t. none of its
          sentences contain words that aren't in train
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

# -----
# Building and testing the model
# -----

# Records bigram and unigram empirical probabilities into a dictionary
def itp_model(training_data):
    freq_dict = frequencies(training_data)
    prob_dict = defaultdict(float)
    # (1) Obtain bigram empirical conditional probabilities
    for context in freq_dict['fw']:
        # (a) Get dict of freqs of words that occurred after context
        next_word_dict = freq_dict['fw'][context]
        # (b) Calculate total frequency of context, by summing the freqs
        #     with which each word occurred after it
        context_freq = sum(next_word_dict.values())
        # (c) Record empirical cond. prob. of each word after context
        for word in next_word_dict:
            bigram_freq = next_word_dict[word]
            prob_dict[(context, word)] = bigram_freq / context_freq
    # (2/i) Get total number of bigram tokens in data (by summing the
    # frequencies with which each context occurred before each word)
    total_freq = 0
    for word in freq_dict['bw']:
        # v--[change!]--v
        for context in freq_dict['bw'][word]:
            total_freq += freq_dict['bw'][word][context]
    # (2/ii) Obtain unigram empirical probabilities
    for word in freq_dict['bw']:
        # (a) Get dict of freqs of contexts that occurred before word
        context_dict = freq_dict['bw'][word]
        # (b) v--[change!]--v Calculate frequency of word
        word_freq = sum(context_dict.values())
        # (c) Record empirical probability of word
        prob_dict[word] = word_freq / total_freq
    return prob_dict

# Calculates the perplexity of model on test data with given weights
def perplexity(test_data, model, bigr_wt, unigr_wt):
    rate = 1 / sum(len(sentence) + 1 for sentence in test_data)
    cross_entropy = sum(log2(1 / itp_prob(sentence, model, bigr_wt, unigr_wt))
                        for sentence in test_data)
    return exp2(cross_entropy * rate)

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

# Records forward and backward cooccurrence frequencies into a dict
def frequencies(training_data):
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

# Calculates interpolated probability of sentence according to model and weights
def itp_prob(sentence, model, bigr_wt, ungr_wt):
    prob = 1
    for bigram in bigrams_list(sentence):
        weighted_bigram_prob  = model[bigram] * bigr_wt
        weighted_unigram_prob = model[bigram[1]] * ungr_wt
        prob *= weighted_bigram_prob + weighted_unigram_prob
    return prob