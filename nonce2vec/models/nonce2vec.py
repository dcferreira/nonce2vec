# -*- encoding: utf-8 -*-
# pylint: skip-file
"""Nonce2Vec model.

A modified version of gensim.Word2Vec.
"""

import logging
import os
from collections import defaultdict, OrderedDict
from enum import Enum
from pathlib import Path
from typing import Optional, Union, Sequence, Dict, Any, Callable, List, Set

import numpy as np
import numpy.typing as npt
from gensim.models.word2vec import Word2Vec
from gensim.utils import keep_vocab_item
from scipy.special import expit
from six import iteritems

__all__ = "Nonce2Vec"

from nonce2vec.models.informativeness import Informativeness

logger = logging.getLogger(__name__)


class LearningRateFunction(str, Enum):
    EXP = "exp_alpha"
    CWI = "cwi_alpha"
    CST = "cst_alpha"


class CustomTrainingWord2Vec(Word2Vec):
    def __init__(self, *args, n2vmodel: "Nonce2Vec", **kwargs):
        super().__init__(*args, **kwargs)
        self.n2v = n2vmodel

    # noinspection PyMethodOverriding
    @classmethod
    def load(cls, *args, n2vmodel: "Nonce2Vec", **kwargs):
        model = super().load(*args, **kwargs)
        model.n2v = n2vmodel
        return model

    @classmethod
    def from_w2v(cls, w2vmodel: Word2Vec, n2vmodel: "Nonce2Vec"):
        w2vmodel.n2v = n2vmodel
        w2vmodel.__class__ = cls
        return w2vmodel

    def _do_train_job(self, sentences, *args, **kwargs):
        """Train a single batch of sentences.

        Return 2-tuple `(effective word count after ignoring unknown words
        and sentence length trimming, total word count)`.
        """
        tally = train_batch_sg(
            nonce2vec=self.n2v,
            sentences=sentences,
        )
        # noinspection PyProtectedMember
        return tally, self._raw_word_count(sentences)


class Nonce2Vec:
    def __init__(
        self,
        model: Union[Path, CustomTrainingWord2Vec, Word2Vec],
        info_model: Informativeness,
        reload: bool = False,
        epochs: int = 5,
        train_with: LearningRateFunction = LearningRateFunction.CWI,
        lambda_decay: float = 70,
        kappa: int = 1,
        beta: int = 1000,
        sample_decay: float = 1.9,
        window_decay: int = 15,
        sum_only: bool = False,
        reduced: bool = False,
        sum_over_set: bool = False,
        weighted: bool = False,
        train_over_set: bool = False,
        with_stats: bool = False,
        shuffle: bool = False,
    ):
        self.model: CustomTrainingWord2Vec
        if isinstance(model, Path):
            self.model = CustomTrainingWord2Vec.load(os.fspath(model), n2vmodel=self)
        elif isinstance(model, Word2Vec):
            self.model = CustomTrainingWord2Vec.from_w2v(model, self)
        else:
            self.model = model
        self.info_model = info_model
        self.reload = reload
        self.epochs = epochs
        self.train_with = train_with
        self.lambda_decay = lambda_decay
        self.kappa = kappa
        self.beta = beta
        self.sample_decay = sample_decay
        self.window_decay = window_decay
        self.sum_only = sum_only
        self.reduced = reduced
        self.sum_over_set = sum_over_set
        self.weighted = weighted
        self.train_over_set = train_over_set
        self.with_stats = with_stats
        self.shuffle = shuffle

        self.vectors_lockf: npt.NDArray[np.float32] = np.ones(
            len(self.model.wv), dtype=np.float32
        )
        self.neg_labels: npt.NDArray[np.float32] = np.array([])
        if self.model.negative > 0:
            # precompute negative labels optimization for pure-python training
            self.neg_labels = np.zeros(self.model.negative + 1).astype(np.float32)
            self.neg_labels[0] = 1.0

        self.new_nonces: Optional[List[str]] = None
        self._new_nonces_set: Optional[Set[str]] = None

    def add_nonces(self, sentences: Sequence[Sequence[str]]):
        if self.reduced:
            self.build_vocab([sentences[0]], update=True)
        else:
            self.build_vocab(sentences, update=True)
        if not self.sum_only:
            self.model.train(
                sentences, total_examples=self.model.corpus_count, epochs=self.epochs
            )

    def build_vocab(
        self,
        sentences: Sequence[Sequence[str]],
        update: bool = False,
        progress_per: int = 10000,
        keep_raw_vocab: bool = False,
        trim_rule: Optional[Callable[[str], bool]] = None,
        **kwargs,
    ):
        total_words, corpus_count = self.model.scan_vocab(
            sentences, progress_per=progress_per, trim_rule=trim_rule
        )
        self.model.corpus_count = corpus_count
        report_values, pre_exist_words = self.prepare_vocab(
            self.model.hs,
            self.model.negative,
            self.model.wv,
            update=update,
            keep_raw_vocab=keep_raw_vocab,
            trim_rule=trim_rule,
            **kwargs,
        )
        report_values["memory"] = self.model.estimate_memory(
            vocab_size=report_values["num_retained_words"]
        )
        self.prepare_weights(
            pre_exist_words,
            self.model.hs,
            self.model.negative,
            self.model.wv,
            sentences,
            update=update,
            sum_over_set=self.sum_over_set,
            weighted=self.weighted,
            beta=self.beta,
        )

    def recompute_sample_ints(self):
        for w in self.model.wv.key_to_index.keys():
            old_value = float(self.model.wv.get_vecattr(w, "sample_int"))
            self.model.wv.set_vecattr(
                w, "sample_int", int(round(old_value / float(self.sample_decay)))
            )

    def prepare_weights(
        self,
        pre_exist_words,
        hs,
        negative,
        wv,
        sentences,
        # nonce,
        update=False,
        sum_over_set=False,
        weighted=False,
        beta=1000,
    ):
        """Build tables and model weights based on final vocabulary settings."""
        # set initial input/projection and hidden weights
        if not update:
            raise Exception(
                "prepare_weight on Nonce2VecTrainables should "
                "always be used with update=True"
            )
        else:
            self.update_weights(
                pre_exist_words,
                hs,
                negative,
                wv,
                sentences,
                # nonce,
                sum_over_set,
                weighted,
                beta,
            )

    def update_weights(
        self,
        pre_exist_words,
        hs,
        negative,
        wv,
        sentences,
        # nonce,
        sum_over_set=False,
        weighted=False,
        beta=1000,
    ):
        """
        Copy all the existing weights, and reset the weights for the newly
        added vocabulary.
        """
        logger.info("updating layer weights")
        gained_vocab = len(wv) - len(wv.vectors)
        newvectors = np.zeros((gained_vocab, wv.vector_size), dtype=np.float32)
        self.new_nonces = wv.index_to_key[len(wv.vectors) :]
        self._new_nonces_set = list(self.new_nonces)

        # randomize the remaining words
        # FIXME as-is the code is bug-prone. We actually only want to
        # initialize the vector for the nonce, not for the remaining gained
        # vocab. This implies that the system should be run with the same
        # min_count as the pre-trained background model. Otherwise
        # we won't be able to sum as we won't have vectors for the other
        # gained background words
        # if gained_vocab == 0:
        #     raise Exception(
        #         "Nonce word '{}' already in test set and not "
        #         "properly deleted".format(nonce)
        #     )
        for i, nonce in enumerate(self.new_nonces, start=len(wv.vectors)):
            # Initialise to sum
            raw_ctx, filtered_ctx = self.info_model.filter_sum_context(
                sentences, pre_exist_words, nonce
            )
            if sum_over_set:
                raw_ctx = set(raw_ctx)
                filtered_ctx = set(filtered_ctx)
                logger.debug(
                    "Summing over set of context items: {}".format(filtered_ctx)
                )
            ctx_ent_map: Optional[Dict[str, float]] = None
            if weighted:
                logger.debug(
                    "Applying weighted sum"
                )  # Sum over positive cwi words only
                ctx_ent_map = self.info_model.get_ctx_ent_for_weighted_sum(
                    sentences, pre_exist_words, nonce
                )
            if filtered_ctx:
                for w in filtered_ctx:
                    # Initialise to sum
                    if weighted and ctx_ent_map is not None:
                        # hacky reuse of compute_cwi_alpha to compute the
                        # weighted sum with cwi but compensating with
                        # beta for narrow distrib of cwi
                        newvectors[i - len(wv.vectors)] += wv.vectors[
                            wv.key_to_index[w]
                        ] * compute_cwi_alpha(
                            ctx_ent_map[w], kappa=1, beta=beta, alpha=1, min_alpha=0
                        )
                    else:
                        newvectors[i - len(wv.vectors)] += wv.vectors[
                            wv.key_to_index[w]
                        ]
            # If no filtered word remains, sum over everything to get 'some'
            # information
            else:
                logger.warning(
                    "No words left to sum over given filter settings. "
                    "Backtracking to sum over all raw context words"
                )
                for w in raw_ctx:
                    # Initialise to sum
                    newvectors[i - len(wv.vectors)] += wv.vectors[wv.key_to_index[w]]

        # Raise an error if an online update is run before initial training on
        # a corpus
        if not len(wv.vectors):
            raise RuntimeError(
                "You cannot do an online vocabulary-update of a "
                "model which has no prior vocabulary. First "
                "build the vocabulary of your model with a "
                "corpus before doing an online update."
            )

        wv.vectors = np.vstack([wv.vectors, newvectors])
        if negative:
            self.model.syn1neg = np.vstack(
                [
                    self.model.syn1neg,
                    np.zeros((gained_vocab, self.model.layer1_size), dtype=np.float32),
                ]
            )
        wv.vectors_norm = None

        # do not suppress learning for already learned words
        self.vectors_lockf = np.ones(len(wv), dtype=np.float32)

    def prepare_vocab(
        self,
        hs,
        negative,
        wv,
        update=False,
        keep_raw_vocab=False,
        trim_rule=None,
        min_count=None,
        sample=None,
        dry_run=False,
    ):
        min_count = min_count or self.model.min_count
        sample = sample or self.model.sample
        drop_total = drop_unique = 0

        if not update:
            raise Exception("Nonce2Vec can only update a pre-existing vocabulary")
        logger.info("Updating model with new vocabulary")
        new_total = pre_exist_total = 0
        # New words and pre-existing words are two separate lists
        new_words = []
        pre_exist_words = []
        for word, v in iteritems(self.model.raw_vocab):
            # Update count of all words already in vocab
            if word in wv:
                pre_exist_words.append(word)
                pre_exist_total += v
                if not dry_run:
                    wv.set_vecattr(word, "count", wv.get_vecattr(word, "count") + v)
            else:
                # For new words, keep the ones above the min count
                # AND the nonce (regardless of count)
                if (
                    keep_vocab_item(word, v, min_count, trim_rule=trim_rule)
                    or word in self._new_nonces_set
                ):
                    new_words.append(word)
                    new_total += v
                    if not dry_run:
                        wv.key_to_index[word] = len(wv)
                        wv.index_to_key.append(word)
                        wv.set_vecattr(word, "count", v)
                else:
                    drop_unique += 1
                    drop_total += v
        original_unique_total = len(pre_exist_words) + len(new_words) + drop_unique
        pre_exist_unique_pct = (
            len(pre_exist_words) * 100 / max(original_unique_total, 1)
        )
        new_unique_pct = len(new_words) * 100 / max(original_unique_total, 1)
        logger.info(
            "New added %i unique words (%i%% of original %i) "
            "and increased the count of %i pre-existing words "
            "(%i%% of original %i)",
            len(new_words),
            new_unique_pct,
            original_unique_total,
            len(pre_exist_words),
            pre_exist_unique_pct,
            original_unique_total,
        )
        retain_words = new_words + pre_exist_words
        retain_total = new_total + pre_exist_total

        # Precalculate each vocabulary item's threshold for sampling
        if not sample:
            # no words downsampled
            threshold_count = retain_total
        # Only retaining one subsampling notion from original gensim implementation
        else:
            threshold_count = sample * retain_total

        downsample_total, downsample_unique = 0, 0
        for w in retain_words:
            v = wv.get_vecattr(w, "count")
            word_probability = (np.sqrt(v / threshold_count) + 1) * (
                threshold_count / v
            )
            if word_probability < 1.0:
                downsample_unique += 1
                downsample_total += word_probability * v
            else:
                word_probability = 1.0
                downsample_total += v
            if not dry_run:
                wv.set_vecattr(
                    w, "sample_int", np.uintc(round(word_probability * 2**32))
                )

        if not dry_run and not keep_raw_vocab:
            logger.info(
                "deleting the raw counts dictionary of %i items",
                len(self.model.raw_vocab),
            )
            self.model.raw_vocab = defaultdict(int)

        logger.info(
            "sample=%g downsamples %i most-common words", sample, downsample_unique
        )
        logger.info(
            "downsampling leaves estimated %i word corpus " "(%.1f%% of prior %i)",
            downsample_total,
            downsample_total * 100.0 / max(retain_total, 1),
            retain_total,
        )

        # return from each step: words-affected, resulting-corpus-size,
        # extra memory estimates
        report_values: Dict[str, Any] = {
            "drop_unique": drop_unique,
            "retain_total": retain_total,
            "downsample_unique": downsample_unique,
            "downsample_total": int(downsample_total),
            "num_retained_words": len(retain_words),
        }

        if self.model.null_word:
            # create null pseudo-word for padding when using concatenative
            # L1 (run-of-words)
            # this word is only ever input – never predicted – so count,
            # huffman-point, etc doesn't matter
            self.model.add_null_word()

        if self.model.sorted_vocab and not update:
            self.model.wv.sort_by_descending_frequency()
        if hs:
            # add info about each word's Huffman encoding
            self.model.create_binary_tree()
        if negative:
            # build the table for drawing random words (for negative sampling)
            self.model.make_cum_table()

        return report_values, pre_exist_words


def compute_cwi_alpha(
    cwi: float, kappa: float, beta: float, alpha: float, min_alpha: float
) -> float:
    x = np.tanh(cwi * beta)
    decay = (np.exp(kappa * (x + 1)) - 1) / (np.exp(2 * kappa) - 1)
    if decay * alpha > min_alpha:
        return decay * alpha
    return min_alpha


def compute_exp_alpha(
    nonce_count: int, lambda_den: float, alpha: float, min_alpha: float
) -> float:
    exp_decay = -(nonce_count - 1) / lambda_den
    if alpha * np.exp(exp_decay) > min_alpha:
        return alpha * np.exp(exp_decay)
    return min_alpha


def train_sg_pair(
    nonce2vec: Nonce2Vec,
    word: str,
    nonce: str,
    context_index: int,
    alpha: float,
    learn_vectors: bool = True,
    learn_hidden: bool = True,
    context_vectors=None,
    context_locks=None,
):
    if context_vectors is None:
        # context_vectors = model.wv.syn0
        context_vectors = nonce2vec.model.wv.vectors

    if context_locks is None:
        # context_locks = model.syn0_lockf
        context_locks = nonce2vec.vectors_lockf

    if word not in nonce2vec.model.wv:
        return
    predict_word_idx = nonce2vec.model.wv.key_to_index[word]  # target word (NN output)

    l1 = context_vectors[context_index]  # input word (NN input/projection layer)
    neu1e = np.zeros(l1.shape)

    # Only train the nonce
    if nonce2vec.model.wv.index_to_key[context_index] == nonce and word != nonce:
        lock_factor = context_locks[context_index]
        if nonce2vec.model.negative:
            # use this word (label = 1) + `negative` other random words not
            # from this sentence (label = 0)
            word_indices = [predict_word_idx]
            while len(word_indices) < nonce2vec.model.negative + 1:
                w = nonce2vec.model.cum_table.searchsorted(
                    nonce2vec.model.random.randint(nonce2vec.model.cum_table[-1])
                )
                if w != predict_word_idx:
                    word_indices.append(w)
            l2b = nonce2vec.model.syn1neg[word_indices]  # 2d matrix, k+1 x layer1_size
            prod_term = np.dot(l1, l2b.T)
            fb = expit(prod_term)  # propagate hidden -> output
            gb = (nonce2vec.neg_labels - fb) * alpha  # vector of error gradients
            # multiplied by the learning rate
            if learn_hidden:
                nonce2vec.model.syn1neg[word_indices] += np.outer(gb, l1)
                # learn hidden -> output
            neu1e += np.dot(gb, l2b)  # save error

        if learn_vectors:
            l1 += neu1e * lock_factor  # learn input -> hidden
            # (mutates model.wv.syn0[word2.index], if that is l1)
    return neu1e


def _get_unique_ctx_ent_tuples(ctx_ent_tuples):
    ctx_ent_dict = OrderedDict()
    for ctx, ent in ctx_ent_tuples:
        if ctx not in ctx_ent_dict:
            ctx_ent_dict[ctx] = ent
        else:
            ctx_ent_dict[ctx] = max(ent, ctx_ent_dict[ctx])
    return [(ctx, ent) for ctx, ent in ctx_ent_dict.items()]


def train_batch_sg(nonce2vec: Nonce2Vec, sentences: Sequence[Sequence[str]]) -> int:
    if nonce2vec.new_nonces is None:
        raise ValueError("Trying to train a batch before setting new nonces!")
    result = 0
    for current_nonce in nonce2vec.new_nonces:
        alpha = nonce2vec.model.alpha  # re-initialize learning rate before each batch
        ctx_ent_tuples = nonce2vec.info_model.filter_and_sort_train_ctx_ent(
            sentences, nonce2vec.model.wv, current_nonce
        )
        if nonce2vec.train_over_set:
            logger.debug("Training over set of context items")
            ctx_ent_tuples = _get_unique_ctx_ent_tuples(ctx_ent_tuples)
        logger.debug("Training on context = {}".format(ctx_ent_tuples))
        nonce_vocab_idx = nonce2vec.model.wv.key_to_index[current_nonce]
        nonce_count = 0
        for ctx_word, cwi in ctx_ent_tuples:
            ctx_vocab_idx = nonce2vec.model.wv.key_to_index[ctx_word]
            nonce_count += 1
            if not nonce2vec.train_with:
                raise Exception(
                    "Unspecified learning rate decay function. "
                    "You must specify a 'train_with' parameter"
                )
            if nonce2vec.train_with == "cwi_alpha":
                alpha = compute_cwi_alpha(
                    cwi,
                    nonce2vec.kappa,
                    nonce2vec.beta,
                    nonce2vec.model.alpha,
                    nonce2vec.model.min_alpha,
                )
                logger.debug(
                    "training on '{}' and '{}' with cwi = {}, b_cwi = {}, "
                    "alpha = {}".format(
                        nonce2vec.model.wv.index_to_key[nonce_vocab_idx],
                        nonce2vec.model.wv.index_to_key[ctx_vocab_idx],
                        round(cwi, 5),
                        round(np.tanh(nonce2vec.beta * cwi), 4),
                        round(alpha, 5),
                    )
                )
            if nonce2vec.train_with == "exp_alpha":
                alpha = compute_exp_alpha(
                    nonce_count,
                    nonce2vec.lambda_decay,
                    nonce2vec.model.alpha,
                    nonce2vec.model.min_alpha,
                )
                logger.debug(
                    "training on '{}' and '{}' with cwi = {}, "
                    "alpha = {}".format(
                        nonce2vec.model.wv.index_to_key[nonce_vocab_idx],
                        nonce2vec.model.wv.index_to_key[ctx_vocab_idx],
                        round(cwi, 5),
                        round(alpha, 5),
                    )
                )
            if nonce2vec.train_with == "cst_alpha":
                alpha = nonce2vec.model.alpha
            train_sg_pair(
                nonce2vec,
                nonce2vec.model.wv.index_to_key[ctx_vocab_idx],
                current_nonce,
                nonce_vocab_idx,
                alpha,
            )
            result += len(ctx_ent_tuples) + 1
    return result
