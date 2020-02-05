#!/usr/bin/env python
from collections import OrderedDict

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('punkt')
nltk.download("stopwords")
from nltk.corpus import stopwords

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

import spacy
import string
import re


SIMPLE_URL_REGEX = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
PARAGRAPH_REGEX = re.compile(r'\n\n+')
CHARS_REGEX = re.compile(r'[^a-zA-Z0-9\s]')

NES_CATEGORIES = {'PERSON': 'person', 'ORG': 'organization', 'GPE': 'place', 'DATE': 'date/period', 'TIME': 'time', 'MONEY': 'money'}
MIN_SENTENCES = 10
SUMM_COEF = 3

class Summarizer:
    def __init__(self):
        # Download the model first (python -m spacy download en_core_web_lg)
        self.__nlp = spacy.load("en_core_web_lg")

    def get_summary(self, text):
        result = {}
        vectors, nes, sentences = self.__get_text_info(text)

        result['entities'] = nes
        summary_sents = sentences
        if len(sentences) > MIN_SENTENCES:
            sorted_scores = self.__get_text_rank_scores(vectors)
            summary_size = len(sentences) // SUMM_COEF
            max_idx = sorted([item[0] for item in sorted_scores[:summary_size]])
            summary_sents = [sentences[idx] for idx in max_idx]

        result['summary'] = '\n'.join(summary_sents)

        return result

    def __clean_text(self, text):
        text = SIMPLE_URL_REGEX.sub('', text)
        text = PARAGRAPH_REGEX.sub('', text)

        return text

    def __get_embeddings_spacy(self, texts):
        docs = self.__nlp.pipe(texts, disable=['tagger', 'parser'])
        vectors = [doc.vector for doc in docs]
        return vectors

    def __get_text_info(self, text):

        cleaned_text = self.__clean_text(text)
        sentences = sent_tokenize(cleaned_text)

        nes = dict()
        sent_docs = list(self.__nlp.pipe(sentences, disable=['parser']))
        for idx, doc in enumerate(sent_docs):
            for ent in doc.ents:
                sent_ents = OrderedDict()
                if ent.text not in sent_ents and ent.label_ in NES_CATEGORIES.keys():
                    sent_ents[ent.text] = NES_CATEGORIES[ent.label_]
                nes[idx] = sent_ents

        vectors = []
        # remove the sentences with no valuable content
        if len(sentences) > MIN_SENTENCES:
            sentences_to_sign = self.__get_significant_sentences(sentences)

            if len(sentences_to_sign) > 0:
                sentences = list(sentences_to_sign.keys())
                vectors = self.__get_embeddings_spacy(list(sentences_to_sign.values()))

        return (vectors, nes, sentences)

    def __get_significant_sentences(self, sentences):
        sentence_to_sign_info = OrderedDict()
        stop_wrds = set(stopwords.words())

        for sentence in sentences:
            words = word_tokenize(sentence)
            no_stop_words = ' '.join([word for word in words if word not in stop_wrds])
            removed_symbols = CHARS_REGEX.sub('', no_stop_words).strip()
            if removed_symbols != '':
                sentence_to_sign_info[sentence] = removed_symbols

        return sentence_to_sign_info


    def __get_text_rank_scores(self, vectors):
        sim_matrix = np.zeros([len(vectors), len(vectors)])

        for i in range(0, len(vectors)):
            for j in range(0, len(vectors)):
                if i != j:
                    sim_matrix[i][j] = cosine_similarity(vectors[i].reshape(1, 300), vectors[j].reshape(1, 300))

        nx_graph = nx.from_numpy_array(sim_matrix)
        scores = nx.pagerank(nx_graph)

        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return sorted_scores




if __name__ == '__main__':

    test_mail = """Hey there,

    Sometimes our families—in an effort to protect us and help us grow—can lead us down paths we would not have chosen for ourselves.

    The good thing is that once we realize this, we can kind of get ourselves back on track and pursue the things we truly want!

    That’s exactly what happened to Sandesh, who wrote to me a while back for advice on how to learn Python:

    ~~~

    I just want to devote a lot of my time on grasping each and every concepts of Python programming language and go ahead on my journey from web developer to data scientist.

    I had to lose about 5 years of life due to my family’s dumb decision. They didn’t let me study Computer Science in my bachelor’s. Now, I am starting on my own. 

    I can devote anything that takes to be a professional Python and Django developer. Getting a solid solution and understanding the problem is always a good practice in programming. It will change my life by being a great, practical programmer.

    ~~~

    I know as developers we kind of have a thing for “wasting time.” We actively search for efficient solutions and optimize our code for runtime.

    Being forced to wait for the result you truly want can be agonizing.

    Can you imagine waiting 5 years for your program to run?!

    But real life doesn’t run like a Python script, no matter how much we may want it to. It may take us years to get the results we want, and there will be setbacks and obstacles in our path all along the way.

    Sandesh has the right idea, though. Even though his family threw a wrench in his plans, all that matters now is that he’s ready to pursue his dream of becoming a Python developer.

    The most important thing is that you stay focused on what you truly want, and to cultivate an environment that will keep you pushing towards that goal every single day.

    If you’re facing an obstacle in your path, a setback on your Python journey, then what can you do to make up for that lost time? How can you maximize your efficiency so that you learn as much as possible, as quickly as possible?

    I’ve got three suggestions.

    The first is to leave unsupportive people out of your goals. Being surrounded by encouraging loved ones is an ideal scenario, but this may not be an option.

    If there are people in your life who don’t understand your desire to become a Python developer, you don’t want to risk getting thrown off track again if they decide to start trouble! Your best chance of success in this case is to keep them as far away from your work as possible.

    Now, I know it’s tough to go it alone on any new venture. You do need a network of people who are willing to help you and who understand why you’re taking on this journey. 

    So my second suggestion is to install a new support network into your life, one of like-minded others. 

    Forums like the PythonistaCafe and the Real Python members-only Slack channel offer you a place to connect with developers who will support and encourage you.

    The last thing I would recommend is to find a dedicated resource you can use to learn and stick with it. 

    We can spend hours stuck in analysis paralysis trying to decide what tools or courses we should try next. Find something that you can come back to every day and keep plugging away on.

    If, like Sandesh, you’re eager to move forward on your journey, then the learning paths on Real Python can help you stay on track: 

    → realpython.com/learning-paths

    Happy Pythoning!

    — Dan Bader"""

    summarizer = Summarizer()
    summary = summarizer.get_summary(test_mail)

    print(summary['entities'])
    print(summary['summary'])

    # print(stopwords.words())
    # dummy_sent = "#   () ~~~~~ >>>>>>>> <<<<<<"
    # print(CHARS_REGEX.sub('', dummy_sent).strip())
