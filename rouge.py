import re
import six
import numpy as np
import collections
from rouge_score import rouge_scorer, scoring

def compute(predictions, references, tokenizer, rouge_types=None, use_agregator=True, ):
	if rouge_types is None:
		rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

	scorer = CustomRouge(rouge_types=rouge_types, tokenizer=tokenizer)
	if use_agregator:
		aggregator = scoring.BootstrapAggregator()
	else:
		scores = []

	for ref, pred in zip(references, predictions):
		score = scorer.score(ref, pred)
		if use_agregator:
			aggregator.add_scores(score)
		else:
			scores.append(score)

	if use_agregator:
		result = aggregator.aggregate()
	else:
		result = {}
		for key in scores[0]:
			result[key] = list(score[key] for score in scores)
	return result

def compute_metrics(eval_preds, tokenizer, data_args):
	preds, labels = eval_preds
	if isinstance(preds, tuple):
		preds = preds[0]
	decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
	if data_args.ignore_pad_token_for_loss:
		# Replace -100 in the labels as we can't decode them.
		labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
	decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
	# Some simple post-processing
	result = compute(predictions=decoded_preds, references=decoded_labels, tokenizer=tokenizer)
	# Extract a few results from ROUGE
	result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

	prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
	result["gen_len"] = np.mean(prediction_lens)
	result = {k: round(v, 4) for k, v in result.items()}
	return result

class CustomRouge(rouge_scorer.RougeScorer) :
	""" 
	https://github.com/google-research/google-research/blob/master/rouge
	google-research의 RougeScorer를 상속.
	한국어 tokenizer를 적용한 rouge score를 계산하도록 custom.
	Args:
		rouge_type (list[str]): 원하는 Rouge Score type list
		tokenizer (PreTrainedTokenizerBase): 모델에 맞는 tokenizer
	"""
	def __init__(self, rouge_types, tokenizer) :
		super(CustomRouge).__init__()
		self.rouge_types = rouge_types
		self.tokenizer = tokenizer

	def score(self, target, prediction):
		"""Calculates rouge scores between the target and prediction.
		Args:
			target: Text containing the target (ground truth) text.
			prediction: Text containing the predicted text.
		Returns:
			A dict mapping each rouge type to a Score object.
		Raises:
			ValueError: If an invalid rouge type is encountered.
		"""

		# Pre-compute target tokens and prediction tokens for use by different
		# types, except if only "rougeLsum" is requested.
		if len(self.rouge_types) == 1 and self.rouge_types[0] == "rougeLsum":
			target_tokens = None
			prediction_tokens = None
		else:
			target_tokens = self.tokenizer.tokenize(target)
			prediction_tokens = self.tokenizer.tokenize(prediction)
		result = {}

		for rouge_type in self.rouge_types:
			if rouge_type == "rougeL":
				# Rouge from longest common subsequences.
				scores = _score_lcs(target_tokens, prediction_tokens)
			elif rouge_type == "rougeLsum":
				# Note: Does not support multi-line text.
				def get_sents(text):
					# Assume sentences are separated by newline.
					sents = six.ensure_str(text).split("\n")
					sents = [x for x in sents if len(x)]
					return sents

				target_tokens_list = [
					self.tokenizer.tokenize(s) for s in get_sents(target)]
				prediction_tokens_list = [
					self.tokenizer.tokenize(s) for s in get_sents(prediction)]
				scores = _summary_level_lcs(target_tokens_list,
											prediction_tokens_list)
						
			elif re.match(r"rouge[0-9]$", six.ensure_str(rouge_type)):
				# Rouge from n-grams.
				n = int(rouge_type[5:])
				if n <= 0:
					raise ValueError("rougen requires positive n: %s" % rouge_type)
				target_ngrams = _create_ngrams(target_tokens, n)
				prediction_ngrams = _create_ngrams(prediction_tokens, n)
				scores = _score_ngrams(target_ngrams, prediction_ngrams)
			else:
				raise ValueError("Invalid rouge type: %s" % rouge_type)
			result[rouge_type] = scores

		return result

def _create_ngrams(tokens, n):
	"""Creates ngrams from the given list of tokens.
	Args:
		tokens: A list of tokens from which ngrams are created.
		n: Number of tokens to use, e.g. 2 for bigrams.
	Returns:
		A dictionary mapping each bigram to the number of occurrences.
	"""
	ngrams = collections.Counter()
	for ngram in (tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)):
		ngrams[ngram] += 1
	return ngrams


def _score_lcs(target_tokens, prediction_tokens):
	"""Computes LCS (Longest Common Subsequence) rouge scores.
	Args:
		target_tokens: Tokens from the target text.
		prediction_tokens: Tokens from the predicted text.
	Returns:
		A Score object containing computed scores.
	"""

	if not target_tokens or not prediction_tokens:
		return scoring.Score(precision=0, recall=0, fmeasure=0)

	# Compute length of LCS from the bottom up in a table (DP appproach).
	lcs_table = _lcs_table(target_tokens, prediction_tokens)
	lcs_length = lcs_table[-1][-1]

	precision = lcs_length / len(prediction_tokens)
	recall = lcs_length / len(target_tokens)
	fmeasure = scoring.fmeasure(precision, recall)

	return scoring.Score(precision=precision, recall=recall, fmeasure=fmeasure)

def _lcs_table(target_tokens, prediction_tokens):
	"""Create 2-d LCS score table."""
	rows = len(target_tokens)
	cols = len(prediction_tokens)
	lcs_table = [[0] * (cols + 1) for _ in range(rows + 1)]
	for i in range(1, rows + 1):
		for j in range(1, cols + 1):
			if target_tokens[i - 1] == prediction_tokens[j - 1]:
				lcs_table[i][j] = lcs_table[i - 1][j - 1] + 1
			else:
				lcs_table[i][j] = max(lcs_table[i - 1][j], lcs_table[i][j - 1])
	return lcs_table


def _backtrack_norec(t, target_tokens, prediction_tokens):
	"""Read out LCS."""
	i = len(target_tokens)
	j = len(prediction_tokens)
	lcs = []
	while i > 0 and j > 0:
		if target_tokens[i - 1] == prediction_tokens[j - 1]:
			lcs.insert(0, i-1)
			i -= 1
			j -= 1
		elif t[i][j - 1] > t[i - 1][j]:
			j -= 1
		else:
			i -= 1
	return lcs


def _summary_level_lcs(target_tokens_list, prediction_tokens_list):
	"""ROUGE: Summary-level LCS, section 3.2 in ROUGE paper.
	Args:
	ref_sent: list of tokenized reference sentences
	can_sent: list of tokenized candidate sentences
	Returns:
	summary level ROUGE score
	"""
	if not target_tokens_list or not prediction_tokens_list:
		return scoring.Score(precision=0, recall=0, fmeasure=0)

	m = sum(map(len, target_tokens_list))
	n = sum(map(len, prediction_tokens_list))
	if not n or not m:
		return scoring.Score(precision=0, recall=0, fmeasure=0)

	# get token counts to prevent double counting
	token_cnts_r = collections.Counter()
	token_cnts_c = collections.Counter()
	for s in target_tokens_list:
		# s is a list of tokens
		token_cnts_r.update(s)
	for s in prediction_tokens_list:
		token_cnts_c.update(s)

	hits = 0
	for target_tokens in target_tokens_list:
		lcs = _union_lcs(target_tokens, prediction_tokens_list)
		# Prevent double-counting:
		# The paper describes just computing hits += len(_union_lcs()),
		# but the implementation prevents double counting. We also
		# implement this as in version 1.5.5.
		for t in lcs:
			if token_cnts_c[t] > 0 and token_cnts_r[t] > 0:
				hits += 1
				token_cnts_c[t] -= 1
				token_cnts_r[t] -= 1

	recall = hits / m
	precision = hits / n
	fmeasure = scoring.fmeasure(precision, recall)
	return scoring.Score(precision=precision, recall=recall, fmeasure=fmeasure)


def _union_lcs(target_tokens, prediction_tokens_list):
	"""Find union LCS between a ref sentence and list of candidate sentences.
	Args:
	ref: list of tokens
	c_list: list of list of indices for LCS into reference summary
	Returns:
	List of tokens in ref representing union LCS.
	"""
	lcs_list = [lcs_ind(target_tokens, prediction_tokens) for prediction_tokens in prediction_tokens_list]
	return [target_tokens[i] for i in _find_union(lcs_list)]


def _find_union(lcs_list):
	"""Finds union LCS given a list of LCS."""
	return sorted(list(set().union(*lcs_list)))


def lcs_ind(target_tokens, prediction_tokens):
	"""Returns one of the longest lcs."""
	t = _lcs_table(target_tokens, prediction_tokens)
	return _backtrack_norec(t, target_tokens, prediction_tokens)


def _score_ngrams(target_ngrams, prediction_ngrams):
	"""Compute n-gram based rouge scores.
	Args:
	target_ngrams: A Counter object mapping each ngram to number of
		occurrences for the target text.
	prediction_ngrams: A Counter object mapping each ngram to number of
		occurrences for the prediction text.
	Returns:
	A Score object containing computed scores.
	"""

	intersection_ngrams_count = 0
	for ngram in six.iterkeys(target_ngrams):
		intersection_ngrams_count += min(target_ngrams[ngram], prediction_ngrams[ngram])
	target_ngrams_count = sum(target_ngrams.values())
	prediction_ngrams_count = sum(prediction_ngrams.values())

	precision = intersection_ngrams_count / max(prediction_ngrams_count, 1)
	recall = intersection_ngrams_count / max(target_ngrams_count, 1)
	fmeasure = scoring.fmeasure(precision, recall)

	return scoring.Score(precision=precision, recall=recall, fmeasure=fmeasure)