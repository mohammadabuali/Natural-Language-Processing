import matplotlib.pyplot as plt
import nltk
from nltk.corpus import brown
import numpy as np
import re
import copy

START_WORD = START_TAG = '**START**'
STOP_WORD = STOP_TAG = '**END**'
CAP_A = 'A'
CAP_Z = 'Z'
ZERO = '0'
NINE = '9'
DOLLAR_SYMBOL = '$'
PERIOD = '.'
DASH = '-'
APOSTROPHE = '\''
COMMA = ','
START_WITH_CAPITAL_LETTER = 'startWithCapital'
ENDS_WITH_ED = 'endsWithEd'
CONTAINS_ONLY_DIGITS = 'containsOnlyDigits'
CONTAINS_LETTER_APOSTROPHE = 'containsLetterAndAppostrophie'
CONTAINS_LETTERS_DASH = 'containsLettersAndDash'
CONTAINS_DIGITS_LETTER_DASH = 'containsDigitsLettersDash'
CONTAINS_DIGITS_COMA = 'containsDigitsAndComa'
CONTAINS_DIGITS_PERIOD = 'containsDigitsAndPeriod'
CONTAINS_DIGITS_SYMBOL = 'containsDigitsAndSymbols'
CONTAINS_DIGITS_PERIOD_SYMBOL = 'containsDigitsPeriodSymbol'
CAP_PERIOD = 'CapPeriod'


def cap_period(word):
    if len(word) > 2:
        if word[0].isupper() and word[-1] == PERIOD:
            return True
    return False


def starts_with_cap(word):
    return word[0].isupper()


def ends_with_ed(word):
    if len(word) > 2:
        if word[-2] == 'e' and word[-1] == 'd':
            return True
    return False


def conatins_letters_and_apostraphe(word):
    for i in word:
        if ord(i) >= ord(CAP_A) and ord(i) <= ord(CAP_Z) or i == APOSTROPHE:
            continue
        return False
    return True


def conatins_letters_and_dash(word):
    for i in range(len(word)):
        if ord(word[i]) >= ord(CAP_A) and ord(word[i]) <= ord(CAP_Z) or word[i] == DASH:
            continue
        return False
    return True


def contains_only_digits(word):
    return word.isdigit()


def contains_digits_dash_letters(word):
    for i in range(len(word)):
        if ord(word[i]) >= ord(CAP_A) and ord(word[i]) <= ord(CAP_Z) or word[i] == DASH or word[i].isdigit():
            continue
        return False
    return True


def conatins_numbers_and_comma(word):
    for i in range(len(word)):
        if word[i].isdigit() or word[i] == COMMA:
            continue
        return False
    return True


def conatins_numbers_and_period(word):
    for i in range(len(word)):
        if word[i].isdigit() or word[i] == PERIOD:
            continue
        return False
    return True

def conatins_numbers_and_dollar(word):
    for i in range(len(word)):
        if word[i].isdigit() or word[i] == DOLLAR_SYMBOL:
            continue
        return False
    return True


def conatins_numbers_period_and_dollar(word):
    for i in range(len(word)):
        if word[i].isdigit() or word[i] == DOLLAR_SYMBOL or word[i] == PERIOD:
            continue
        return False
    return True


def pseudolizer(word, low_frequent, unknown):
    if word in low_frequent or word in unknown:
        if cap_period(word):
            return CAP_PERIOD
        if starts_with_cap(word):
            return START_WITH_CAPITAL_LETTER
        if ends_with_ed(word):
            return ENDS_WITH_ED
        if contains_only_digits(word):
            return CONTAINS_ONLY_DIGITS
        if conatins_letters_and_apostraphe(word):
            return CONTAINS_LETTER_APOSTROPHE
        if conatins_letters_and_dash(word):
            return CONTAINS_LETTERS_DASH
        if contains_digits_dash_letters(word):
            return CONTAINS_DIGITS_LETTER_DASH
        if conatins_numbers_and_comma(word):
            return CONTAINS_DIGITS_COMA
        if conatins_numbers_and_period(word):
            return CONTAINS_DIGITS_PERIOD
        if conatins_numbers_and_dollar(word):
            return CONTAINS_DIGITS_SYMBOL
        if conatins_numbers_period_and_dollar(word):
            return CONTAINS_DIGITS_PERIOD_SYMBOL
    return word


def get_data():
    data = brown.tagged_sents(categories='news')
    data = [[[word, re.split(r'\+|-', tag)[0]] if tag != '--' else [word, tag] for word, tag in sentence]
            for sentence in data]
    index = round(0.9 * len(data))
    test_data = copy.deepcopy(data[index:])

    for sentence_idx, sentence in enumerate(data):
        sentence.insert(0, [START_WORD, START_TAG])
        sentence.append([STOP_WORD, STOP_TAG])

    concatenated_data = np.concatenate(data)

    words = np.unique(concatenated_data[:, 0])
    pos = np.unique(concatenated_data[:, 1])

    pos2i = {pos: i for (i, pos) in enumerate(pos)}
    word2i = {word: i for (i, word) in enumerate(words)}

    train_data = data[:index]
    return train_data, test_data, pos, words, pos2i, word2i


def baseline_mle(train_data):
    train_data = np.concatenate(train_data)
    tags_dict = dict()
    for word in np.unique(train_data[:, 0]):
        indices = np.where(train_data[:, 0] == word)
        unique, counts = np.unique(train_data[indices][:, 1], return_counts=True)
        tags_dict[word] = unique[np.argmax(counts)]
    return tags_dict


def compute_error_for_baseline(tags_dict, test_data):
    test_data = np.concatenate(test_data)
    known_words_error_counter = unknown_words_error_counter = known_words_counter = 0

    for word, tag in test_data:
        if word not in tags_dict:
            predicted_tag = 'NN'
            if tag != predicted_tag:
                unknown_words_error_counter += 1
        else:
            predicted_tag = tags_dict[word]
            known_words_counter += 1
            if tag != predicted_tag:
                known_words_error_counter += 1

    all_error_rate = (known_words_error_counter + unknown_words_error_counter) / test_data.shape[0]
    known_words_error_rate = known_words_error_counter / known_words_counter
    unknown_words_error_rate = unknown_words_error_counter / (test_data.shape[0] - known_words_counter)
    return known_words_error_rate, unknown_words_error_rate, all_error_rate


class HMM:
    def __init__(self, train_data, test_data, pos, words, pos2i, word2i):
        self.train_data = train_data
        self.test_data = test_data
        self.training_set_words = np.unique(np.concatenate(self.train_data)[:, 0])
        self.pos_tags = pos
        self.words = words
        self.pos_size = len(pos)
        self.words_size = len(words)

        self.pos2i = pos2i
        self.word2i = word2i
        self.unknown_words = self.find_unknown_words()
        self.low_freq_words = self.find_low_frequency_known_words()
        self.transition, self.emission = self.compute_transition_and_emission()

    def compute_transition_and_emission(self):
        transition = np.zeros((self.pos_size, self.pos_size))
        emission = np.zeros((self.pos_size, self.words_size))
        tags_counter = np.zeros((self.pos_size, 1))

        for sentence in self.train_data:
            for idx, word_tag in enumerate(sentence):
                word, tag = word_tag
                tag_index = self.pos2i[tag]
                word_index = self.word2i[word]
                if idx != len(sentence) - 1:
                    next_tag_index = self.pos2i[sentence[idx + 1][1]]
                    transition[tag_index, next_tag_index] += 1
                emission[tag_index, word_index] += 1
                tags_counter[tag_index][0] += 1

        out1 = np.zeros_like(transition)
        out2 = np.zeros_like(emission)
        np.divide(emission, tags_counter, out=out2, where=tags_counter != 0)
        tags_counter[self.pos2i[STOP_WORD]][0] = 0
        np.divide(transition, tags_counter, out=out1, where=tags_counter != 0)
        return out1, out2

    def Viterbi(self, sentence):
        words_number = len(sentence)
        probability_table = np.zeros((words_number, self.pos_size))
        backpointer_table = np.zeros((words_number, self.pos_size)).astype(int)

        # filling the tables for the base case
        probability_table[0] = self.transition[self.pos2i[START_TAG]] * self.emission.T[self.word2i[sentence[0]]]

        # filling the rest of the tables
        for k in range(1, words_number):

            # in case that the word is unknown we implemented the first choice that described in the forum:
            # "complete the viterbi probabilties table with 0's and arbtirary pointers (as you choose arbitrary argmax)
            # starting from the k's location. When extracting the tags it will result in arbitrary tags for all words
            # starting from the w_k."
            if sentence[k] not in self.training_set_words:
                y_indices = [0] * words_number
                out = ['NN'] * words_number
                index = int(np.argmax(probability_table[k-1]))
                y_indices[k-1] = index
                out[k-1] = self.pos_tags[index]
                for j in range(k - 2, -1, -1):
                    y_indices[j] = backpointer_table[j + 1, y_indices[j + 1]]
                    out[j] = self.pos_tags[y_indices[j]]

                # return optimal tags sequence
                return np.array(out)


            prob_values = []
            word_idx = self.word2i[sentence[k]]
            for idx in range(self.pos_size):
                prob = probability_table[k - 1] * self.transition.T[idx] * self.emission.T[word_idx][idx]
                prob_values.append(prob)

            probability_table[k] = np.max(prob_values, axis=1)
            backpointer_table[k] = np.argmax(prob_values, axis=1)

        # getting the indices of the optimal tags sequence
        tag_sequence_indices = np.zeros(words_number).astype(int)
        tag_sequence = [''] * words_number
        index = np.argmax(probability_table[-1])
        tag_sequence_indices[-1] = index
        tag_sequence[-1] = self.pos_tags[index]

        for i in range(words_number - 2, -1, -1):
            index = backpointer_table[i+1, tag_sequence_indices[i+1]]
            tag_sequence_indices[i] = index
            tag_sequence[i] = self.pos_tags[index]

        # return optimal tags sequence
        return tag_sequence

    def find_unknown_words(self):
        unknown_words = dict()
        for element in self.test_data:
            element = np.array(element)
            sentence = element[:, 0]

            for index, word in enumerate(sentence):
                if word not in self.training_set_words:
                    if word not in unknown_words.keys():
                        unknown_words[word] = 1
                    else:
                        unknown_words[word] += 1
        return unknown_words

    def find_low_frequency_known_words(self):
        known_words = dict()
        low_freq_words = dict()
        for element in self.test_data:
            element = np.array(element)
            sentence = element[:, 0]

            for index, word in enumerate(sentence):
                if word in self.training_set_words:
                    if word not in known_words.keys():
                        known_words[word] = 1
                    else:
                        known_words[word] += 1
        for key in known_words:
            if known_words[key] < 5:
                low_freq_words[key] = known_words[key]

        return low_freq_words


    def compute_error(self):
        known_words_counter = unknown_words_counter = known_words_error_counter = unknown_words_error_counter = 0
        for element in self.test_data:
            element = np.array(element)
            sentence = element[:, 0]
            tags = element[:, 1]

            predicted_tags = self.Viterbi(sentence)
            correct_tags = predicted_tags == tags

            for index, word in enumerate(sentence):
                if not correct_tags[index]:
                    if word in self.training_set_words:
                        known_words_error_counter += 1
                    else:
                        unknown_words_error_counter += 1
                if word in self.training_set_words:
                    known_words_counter += 1
                else:
                    unknown_words_counter += 1

        all_error_rate = (known_words_error_counter + unknown_words_error_counter) / \
                         (known_words_counter + unknown_words_counter)
        known_words_error_rate = known_words_error_counter / known_words_counter
        unknown_words_error_rate = unknown_words_error_counter / unknown_words_counter
        return known_words_error_rate, unknown_words_error_rate, all_error_rate


class HMMAddOneSmoothing:
    def __init__(self, train_data, test_data, pos, words, pos2i, word2i):
        self.train_data = train_data
        self.test_data = test_data
        self.training_set_words = np.unique(np.concatenate(self.train_data)[:, 0])

        self.pos_tags = pos
        self.words = words
        self.pos_size = len(pos)
        self.words_size = len(words)

        self.pos2i = pos2i
        self.word2i = word2i
        self.unknown_words = self.find_unknown_words()
        self.low_freq_words = self.find_low_frequency_known_words()
        self.transition, self.emission = self.compute_transition_and_emission()

    def compute_transition_and_emission(self):
        transition = np.ones((self.pos_size, self.pos_size))
        emission = np.ones((self.pos_size, self.words_size))
        tags_counter = np.zeros((self.pos_size, 1))

        for sentence in self.train_data:
            for idx, word_tag in enumerate(sentence):
                word, tag = word_tag
                tag_index = self.pos2i[tag]
                word_index = self.word2i[word]
                if idx != len(sentence) - 1:
                    next_tag_index = self.pos2i[sentence[idx + 1][1]]
                    transition[tag_index, next_tag_index] += 1
                emission[tag_index, word_index] += 1
                tags_counter[tag_index][0] += 1
        out1 = np.divide(transition, tags_counter + self.words_size)
        out2 = np.divide(emission, tags_counter + self.pos_size)
        return out1, out2


    def Viterbi(self, sentence):
        words_number = len(sentence)
        probability_table = np.zeros((words_number, self.pos_size))
        backpointer_table = np.zeros((words_number, self.pos_size)).astype(int)

        # filling the tables for the base case
        probability_table[0] = self.transition[self.pos2i[START_TAG]] * self.emission.T[self.word2i[sentence[0]]]

        # filling the rest of the tables
        for k in range(1, words_number):
            prob_values = []
            word_idx = self.word2i[sentence[k]]
            for idx in range(self.pos_size):
                prob = probability_table[k - 1] * self.transition.T[idx] * self.emission.T[word_idx][idx]
                prob_values.append(prob)

            probability_table[k] = np.max(prob_values, axis=1)
            backpointer_table[k] = np.argmax(prob_values, axis=1)

        # getting the indices of the optimal tags sequence
        tag_sequence_indices = np.zeros(words_number).astype(int)
        tag_sequence = [''] * words_number
        index = np.argmax(probability_table[-1])
        tag_sequence_indices[-1] = index
        tag_sequence[-1] = self.pos_tags[index]

        for i in range(words_number - 2, -1, -1):
            index = backpointer_table[i+1, tag_sequence_indices[i+1]]
            tag_sequence_indices[i] = index
            tag_sequence[i] = self.pos_tags[index]

        # return optimal tags sequence
        return tag_sequence

    def find_unknown_words(self):
        unknown_words = dict()
        for element in self.test_data:
            element = np.array(element)
            sentence = element[:, 0]

            for index, word in enumerate(sentence):
                if word not in self.training_set_words:
                    if word not in unknown_words.keys():
                        unknown_words[word] = 1
                    else:
                        unknown_words[word] += 1
        return unknown_words

    def find_low_frequency_known_words(self):
        known_words = dict()
        low_freq_words = dict()
        for element in self.test_data:
            element = np.array(element)
            sentence = element[:, 0]

            for index, word in enumerate(sentence):
                if word in self.training_set_words:
                    if word not in known_words.keys():
                        known_words[word] = 1
                    else:
                        known_words[word] += 1
        for key in known_words:
            if known_words[key] < 5:
                low_freq_words[key] = known_words[key]

        return low_freq_words

    def compute_error(self):
        known_words_counter = unknown_words_counter = known_words_error_counter = unknown_words_error_counter = 0
        for element in self.test_data:
            element = np.array(element)
            sentence = element[:, 0]
            tags = element[:, 1]
            predicted_tags = self.Viterbi(sentence)
            correct_tags = predicted_tags == tags

            for index, word in enumerate(sentence):
                if not correct_tags[index]:
                    if word in self.training_set_words:
                        known_words_error_counter += 1
                    else:
                        unknown_words_error_counter += 1
                if word in self.training_set_words:
                    known_words_counter += 1
                else:
                    unknown_words_counter += 1

        all_error_rate = (known_words_error_counter + unknown_words_error_counter) / \
                         (known_words_counter + unknown_words_counter)
        known_words_error_rate = known_words_error_counter / known_words_counter
        unknown_words_error_rate = unknown_words_error_counter / unknown_words_counter
        return known_words_error_rate, unknown_words_error_rate, all_error_rate


def get_data_pseudo(low_freq_words, unknown_words):
    data = brown.tagged_sents(categories='news')
    data = [[[pseudolizer(word, low_freq_words, unknown_words), re.split(r'\+|-', tag)[0]] if tag != '--' else
             [pseudolizer(word, low_freq_words, unknown_words), tag] for word, tag in sentence]
            for sentence in data]
    index = round(0.9 * len(data))
    test_data = copy.deepcopy(data[index:])

    for sentence_idx, sentence in enumerate(data):
        sentence.insert(0, [START_WORD, START_TAG])
        sentence.append([STOP_WORD, STOP_TAG])

    concatenated_data = np.concatenate(data)

    words = np.unique(concatenated_data[:, 0])
    pos = np.unique(concatenated_data[:, 1])

    pos2i = {pos: i for (i, pos) in enumerate(pos)}
    word2i = {word: i for (i, word) in enumerate(words)}

    train_data = data[:index]
    return train_data, test_data, pos, words, pos2i, word2i


def draw_plot(known_error, unknown_error, all_error, hmm_known_error, hmm_unknown_error, hmm_all_error,
              lap_hmm_known_error, lap_hmm_unknown_error, lap_hmm_all_error,
              pse_hmm_known_error, pse_hmm_unknown_error, pse_hmm_all_error,
              pse_lap_hmm_known_error, pse_lap_hmm_unknown_error, pse_lap_hmm_all_error,):
    n_groups = 5
    known_err = (known_error, hmm_known_error, lap_hmm_known_error, pse_hmm_known_error, pse_lap_hmm_known_error)
    unknown_err = (unknown_error, hmm_unknown_error, lap_hmm_unknown_error, pse_hmm_unknown_error,
                   pse_lap_hmm_unknown_error)
    all_err = (all_error, hmm_all_error, lap_hmm_all_error, pse_hmm_all_error, pse_lap_hmm_all_error)

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.2
    opacity = 0.8

    rects1 = plt.bar(index, known_err, bar_width,
                     alpha=opacity,
                     color='b',
                     label='Known Error')

    rects2 = plt.bar(index + bar_width, unknown_err, bar_width,
                     alpha=opacity,
                     color='g',
                     label='Unknown Error')

    rects3 = plt.bar(index + 2 * bar_width, all_err, bar_width,
                     alpha=opacity,
                     color='r',
                     label='All Error')

    plt.xlabel('Method')
    plt.ylabel('Error Rate')
    plt.xticks(index + bar_width, ('Basic', 'HMM', 'Laplace', 'Pseudo \n Words', 'Laplace \n + \n Pseudo Words'))
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    nltk.download('brown')
    train_data, test_data, pos, words, pos2i, word2i = get_data()

    tags_dict = baseline_mle(train_data)
    known_words_error_rate, unknown_words_error_rate, all_error_rate = compute_error_for_baseline(tags_dict, test_data)
    print('Basic MLE :')
    print(' known word error rate : %s\n unknown word error rate : %s\n all words error rate : %s\n' %
          (known_words_error_rate, unknown_words_error_rate, all_error_rate))
    tagger = HMM(train_data, test_data, pos, words, pos2i, word2i)
    known1, unknown1, all1 = tagger.compute_error()
    # print(tagger.compute_error())
    print('HMM :')
    print(' known word error rate : %s\n unknown word error rate : %s\n all words error rate : %s\n' %
          (known1, unknown1, all1))
    tagger = HMMAddOneSmoothing(train_data, test_data, pos, words, pos2i, word2i)
    known2, unknown2, all2 = tagger.compute_error()
    # print(tagger.compute_error())
    print('Laplace Smoothing :')
    print(' known word error rate : %s\n unknown word error rate : %s\n all words error rate : %s\n' %
          (known2, unknown2, all2))
    train_data, test_data, pos, words, pos2i, word2i = get_data_pseudo(tagger.low_freq_words.keys(),
                                                                       tagger.unknown_words.keys())
    tagger = HMM(train_data, test_data, pos, words, pos2i, word2i)
    known3, unknown3, all3 = tagger.compute_error()
    # print(tagger.compute_error())
    print('Pseudo Words Smoothing :')
    print(' known word error rate : %s\n unknown word error rate : %s\n all words error rate : %s\n' %
          (known3, unknown3, all3))
    tagger = HMMAddOneSmoothing(train_data, test_data, pos, words, pos2i, word2i)
    known4, unknown4, all4 = tagger.compute_error()
    # print(tagger.compute_error())
    print('Laplace + Pseudo Words Smoothing :')
    print(' known word error rate : %s\n unknown word error rate : %s\n all words error rate : %s\n' %
          (known4, unknown4, all4))
    draw_plot(known_words_error_rate, unknown_words_error_rate, all_error_rate, known1, unknown1, all1,
              known2, unknown2, all2, known3, unknown3, all3, known4, unknown4, all4)

    return 0


if __name__ == '__main__':
    main()


