import nltk
import random as random
# nltk.download('punkt')

START_TOKEN = "<s>"
END_TOKEN = "</s>"

def calculate_unigram(sentences):
    unigrams = {}
    unigrams_probablities = {}
    total_words = 0

    for sentence in sentences:
        for word in sentence:
            total_words = total_words + 1
            unigrams[word] = unigrams.get(word,0) + 1
    
    for key, value in unigrams.items():
        unigrams_probablities[key] = value/total_words

    return unigrams, unigrams_probablities

def calculate_bigram(sentences):
    bigrams = {}
    bigrams_probablities = {}

    unigrams, _ = calculate_unigram(sentences)

    for sentence in sentences:
        for i in range(len(sentence) - 1):
            bigrams[(sentence[i], sentence[i+1])] = bigrams.get((sentence[i], sentence[i+1]), 0) + 1
    
    for key, value in bigrams.items():
        bigrams_probablities[key] = value/int(unigrams[key[0]])

    return bigrams, bigrams_probablities

def print_sentences(sentences):
    for sentence in sentences:
        print(sentence)
        print()

def print_dict(dictionary):
    sorted_dict = sorted(dictionary.items(), key=lambda x: x[1], reverse=True) 

    for item in sorted_dict:
        print(item[0], "\t : ", item[1])

def print_most_frequent_dict(dictionary):
    sorted_dict = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)
    maximum_value = sorted_dict[0][1]

    for item in sorted_dict:
        if(item[1] >= int(maximum_value/2)):
            print(item[0], "\t : ", item[1])
        else:
            return

def corpus_preprocessing(corpus):
    corpus = corpus.lower()
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    sentence_tokenized = nltk.sent_tokenize(corpus)

    vocab = set(tokenizer.tokenize(corpus) + [START_TOKEN, END_TOKEN])
    sentences = []

    for sentence in sentence_tokenized:
        sentences.append(tokenizer.tokenize(sentence))
    
    for sentence in sentences:
        sentence.insert(0, START_TOKEN)
        sentence.append(END_TOKEN)

    return sentences, vocab

def unigram_sentence_sampling(uni_grams_probablities, maxlen):
    keys = list(uni_grams_probablities.keys())
    values = list(uni_grams_probablities.values())

    sentence_sample = []
    for i in range(maxlen):
        sample_word = random.choices(keys, weights = values)[0]
        if(sample_word == START_TOKEN):
            i = i-1
            continue
        elif(sample_word == END_TOKEN):
            sentence_sample.append(sample_word)
            break
        else:
            sentence_sample.append(sample_word)

    return sentence_sample

def bigram_sentence_sampling(bi_grams_probablities, start_tok, maxlen):
    keys = list(bi_grams_probablities.keys())
    values = list(bi_grams_probablities.values())

    sentence_sample = [start_tok]

    for i in range(maxlen):
        last_word = sentence_sample[-1]
        last_word_keys = []
        last_word_values = []
        
        for key in keys:
            if(key[0] == last_word):
                last_word_keys.append(key[1])
                last_word_values.append(bi_grams_probablities[key])
        
        sample_word = random.choices(last_word_keys, weights = last_word_values)[0]

        if(sample_word == START_TOKEN):
            continue
        elif(sample_word == END_TOKEN):
            sentence_sample.append(END_TOKEN)
            break
        else:
            sentence_sample.append(sample_word)
        
    return sentence_sample

corpus1 = "i am utkarsh jain."

corpus1_sentences, corpus1_vocab = corpus_preprocessing(corpus1)

corpus1_unigrams, corpus1_unigrams_probablities = calculate_unigram(corpus1_sentences)
corpus1_bigrams, corpus1_bigrams_probablities = calculate_bigram(corpus1_sentences)

maxlen = 40
sampled_sentence = bigram_sentence_sampling(corpus1_bigrams_probablities, START_TOKEN, maxlen)

print(sampled_sentence)