import xml.etree.ElementTree
import random
import math
import numpy as np


def extract_alpha_words(page):
    words = page.text.split()
    alpha_words = list(filter(lambda word: word.isalpha(), words))
    alpha_lower_words = list(map(lambda word: word.lower(), alpha_words))
    return alpha_lower_words


def print_random_words(word_dict):
    print("Random words' sample:")
    words = list(word_dict.keys())
    sample = random.sample(range(len(words)), 10)
    for index in sample:
        print(words[index])


def compute_erank(X):
    N, M = X.shape
    print('N = {}, M = {}'.format(N, M))
    C = X.dot(X.T)
    print('cov_matrix shape is {}'.format(C.shape))
    _, s, _ = np.linalg.svd(C)
    s_sum = np.sum(s)
    p = np.zeros(N)
    for i in range(N):
        p[i] = s[i] / s_sum
    entropy = 0
    for i in range(N):
        entropy -= p[i] * math.log(p[i])
    print('Entropy is = {}'.format(entropy))
    erank = math.exp(entropy)
    return erank


def main():
    dump_file = 'wikipedia_2000_dump.xml'
    tree = xml.etree.ElementTree.parse(dump_file)
    root = tree.getroot()

    pages_words = []
    for page in root:
        pages_words.append(extract_alpha_words(page))
    pages_count = len(pages_words)
    print('Pages count =', pages_count)

    word_dict = {}
    word_count = 0
    for page_words in pages_words:
        for word in page_words:
            if word not in word_dict:
                word_dict[word] = word_count
                word_count += 1
    print('Unique alpha words =', word_count)

    print_random_words(word_dict)

    X = np.zeros((pages_count, word_count), dtype=np.float32)
    for index, page_words in enumerate(pages_words):
        for word in page_words:
            X[index][word_dict[word]] += (1 / len(page_words))
    print('Effective rank =', compute_erank(X))


if __name__ == '__main__':
    main()

