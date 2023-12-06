"""Author: Brandon Trabucco, Copyright 2019.
Functions to help serialize a caption dataset.
"""


import nltk
import numpy as np
import collections


stemmer = nltk.stem.snowball.SnowballStemmer("english")


def sentence_to_ngrams(sentence, n):
    current_grams = collections.defaultdict(int)
    # lemmas = tuple(stemmer.stem(x) for x in nltk.tokenize.word_tokenize(
    #     sentence.strip().lower()))
    lemmas = tuple(x for x in sentence.split())

    for i in range(len(lemmas) - n + 1):
        current_grams[lemmas[i:(i + n)]] += 1
    return current_grams


def load_ngrams(id_to_captions, n):
    id_to_ngrams = {}
    document_frequencies = collections.OrderedDict()
    for image_id, captions in id_to_captions.items():
        id_to_ngrams[image_id] = []
        unique_ngrams = set()
        for caption in captions:
            ngrams = sentence_to_ngrams(caption, n)
            id_to_ngrams[image_id].append(ngrams)
            for gram in ngrams:
                if gram not in document_frequencies:
                    document_frequencies[gram] = 0
                if gram not in unique_ngrams:
                    unique_ngrams.add(gram)
                    document_frequencies[gram] += 1
    return id_to_ngrams, document_frequencies


def tf_idf(id_to_ngrams, document_frequencies, n, image_id, candidate):
    candidate_ngrams = sentence_to_ngrams(candidate, n)
    total_frequency = sum(candidate_ngrams.values())
    if total_frequency == 0:
        return np.zeros(len(document_frequencies))
    num_examples = len(id_to_ngrams)
    tf_idf_weight = []
    for gram in document_frequencies.keys():
        tf_idf_weight.append(candidate_ngrams[gram] / total_frequency * np.log(
            num_examples / document_frequencies[gram]))
    return np.array(tf_idf_weight)


def tf_idf_known(id_to_ngrams, document_frequencies, image_id, reference_ngrams):
    total_frequency = sum(reference_ngrams.values())
    if total_frequency == 0:
        return np.zeros(len(document_frequencies))
    num_examples = len(id_to_ngrams)
    tf_idf_weight = []
    for gram in document_frequencies.keys():
        tf_idf_weight.append(reference_ngrams[gram] / total_frequency * np.log(
            num_examples / document_frequencies[gram]))
    return np.array(tf_idf_weight)


def cider_n(id_to_ngrams, document_frequencies, n, image_id, candidate):
    candidate_tf_idf_weight = tf_idf(id_to_ngrams, document_frequencies, n, 
        image_id, candidate)
    candidate_norm = np.linalg.norm(candidate_tf_idf_weight)
    normalized_candidate_vector = (candidate_tf_idf_weight / 
        candidate_norm if candidate_norm > 0 else candidate_tf_idf_weight)
    cider_n_score = 0.0
    for reference_ngrams in id_to_ngrams[image_id]:
        reference_tf_idf_weight = tf_idf_known(id_to_ngrams, document_frequencies, 
            image_id, reference_ngrams)
        reference_norm = np.linalg.norm(reference_tf_idf_weight)
        normalized_reference_vector = (reference_tf_idf_weight / 
            reference_norm if reference_norm > 0 else reference_tf_idf_weight)
        cider_n_score += normalized_candidate_vector.dot(normalized_reference_vector)
    return cider_n_score / len(id_to_ngrams[image_id])


def cider(list_of_id_to_ngrams, list_of_document_frequencies, list_of_n, 
        image_id, candidate):
    cider_score = 0.0
    for id_to_ngrams, document_frequencies, n in zip(
            list_of_id_to_ngrams, list_of_document_frequencies, list_of_n):
        cider_score += cider_n(id_to_ngrams, document_frequencies, n, 
            image_id, candidate)
    return cider_score / len(list_of_n)


def build_cider_scorer(id_to_captions, max_n):
    list_of_n = list(range(1, max_n + 1))
    list_of_id_to_ngrams = []
    list_of_document_frequencies = []
    for n in list_of_n:
        id_to_ngrams, document_frequencies = load_ngrams(id_to_captions, n)
        list_of_id_to_ngrams.append(id_to_ngrams)
        list_of_document_frequencies.append(document_frequencies)
    def score_function(candidate, image_id):
        return cider(list_of_id_to_ngrams, list_of_document_frequencies, list_of_n, 
            image_id, candidate)
    return score_function

if __name__ == "__main__":


    # id_to_captions = {
    #     0: ["a women riding a black motor cycle.", "a biker on the road."],
    #     1: ["an apple sitting on a table.", "a red piece of fruit on a wooden table."],
    #     2: ["a young boy swinging a bat.", "a picture of a baseball game."],
    #     3: ["a computer at a desk with a mug.", "someone was working in their office."],
    #     4: ["a forest with a trail.", "hikers exploring the outdoors."],
    # }

    id_to_captions = {
        0: ["a women riding a black motor cycle."],
        1: ["an apple sitting on a table."],
        2: ["a young boy swinging a bat."],
        3: ["a computer at a desk with a mug."],
        4: ["a forest with a trail."],
    }



    image_id = 1
    candidate = "a red apple sitting on a wooden table."

    scorer = build_cider_scorer(id_to_captions, 4)
    print("CIDEr score for \"{}\" was {}".format(candidate, scorer(candidate, image_id)))