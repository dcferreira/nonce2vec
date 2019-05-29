"""Files utils."""

import os
import smart_open
import random
import logging

__all__ = ('Samples', 'get_zipped_sentences', 'get_sentences',
           'get_model_path', 'get_input_filepaths')

logger = logging.getLogger(__name__)


def get_input_filepaths(dirpath):
    """Return a list of absolute XML filepaths from a given dirpath.

    List all the files under a specific directory.
    """
    return [os.path.join(dirpath, filename) for filename in
            os.listdir(dirpath) if '.xml' in filename]


def get_model_path(datadir, outputdir, train_mode, alpha, neg, window_size,
                   sample, epochs, min_count, size):
    """Return absolute path to w2v model file.

    Model absolute path is computed from the outputdir and the
    datadir name.
    """
    os.makedirs(outputdir, exist_ok=True)
    return os.path.join(
        outputdir,
        '{}.{}.alpha{}.neg{}.win{}.sample{}.epochs{}.mincount{}.size{}.model'
        .format(os.path.basename(datadir), train_mode, alpha, neg, window_size,
                sample, epochs, min_count, size))


def get_zipped_sentences(datazip):
    """Return generator over sentence split of wikidump for gensim.word2vec.

    datazip should be the absolute path to the wikidump.gzip file.
    """
    for filename in os.listdir(smart_open.smart_open(datazip)):
        with open(filename, 'r', encoding='utf-8') as input_stream:
            for line in input_stream:
                yield line.strip().split()


def get_sentences(data):
    """Return generator over sentence split of wikidata for gensim.word2vec."""
    for filename in os.listdir(data):
        if filename.startswith('.'):
            continue
        with open(os.path.join(data, filename), 'r', encoding='utf-8') as input_stream:
            for line in input_stream:
                yield line.strip().split()


class Samples(object):
    """An iterable class (with generators) for gensim and n2v."""

    def __init__(self, input_data, source, shuffle):
        if source not in ['wiki', 'definitions', 'chimeras']:
            raise Exception('Invalid source parameter \'{}\''.format(source))
        self._source = source
        self._datafile = input_data
        self._shuffle = shuffle

    def _iterate_over_wiki(self):
        with open(self._datafile, 'rt', encoding='utf-8') as input_stream:
            for line in input_stream:
                yield line.strip().split()

    def _iterate_over_definitions(self):
        with open(self._datafile, 'rt', encoding='utf-8') as input_stream:
            if self._shuffle:
                logger.info('Iterating over test set in shuffled order')
                input_stream = list(input_stream)
                random.shuffle(input_stream)
            for line in input_stream:
                fields = line.rstrip('\n').split('\t')
                nonce = fields[0]
                sentence = fields[1].replace('___', nonce).split()
                probe = '{}_true'.format(nonce)
                yield [sentence], nonce, probe

    def _iterate_over_chimeras(self):
        with open(self._datafile, 'rt', encoding='utf-8') as input_stream:
            if self._shuffle:
                logger.info('Iterating over test set in shuffled order')
                input_stream = list(input_stream)
                random.shuffle(input_stream)
            for num, line in enumerate(input_stream):
                nonce = 'chimera_nonce_{}'.format(num+1)
                fields = line.rstrip('\n').split('\t')
                sentences = [[token if token != '___' else nonce for token in sent.strip().split(' ')] for sent in fields[1].split('@@')]
                probes = fields[2].split(',')
                responses = fields[3].split(',')
                yield sentences, nonce, probes, responses

    def __iter__(self):
        if self._source == 'wiki':
            return self._iterate_over_wiki()
        if self._source == 'definitions':
            return self._iterate_over_definitions()
        if self._source == 'chimeras':
            return self._iterate_over_chimeras()
        raise Exception('Invalid source parameter \'{}\''.format(self._source))
