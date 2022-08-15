import pickle
import numpy as np


def tent_map(x):
	if x < .5:
		return 2*x
	else:
		return 2-2*x


def calc_word_entropy(doc_word_counts, total_word_count):
	"""
	This function should return 1 for a perfectly spread-out distribution, and, I dunno... 0 if in only one document?
	"""
	s_vals = np.array(doc_word_counts) / total_word_count
	entropy = -np.sum(s_vals * np.nan_to_num(np.log2(s_vals))) / np.log2(len(doc_word_counts))
	return tent_map(entropy)  # 4 * entropy * (1 - entropy)


filepath = '/Users/raymondbaranski/Desktop/some_word_data/word_data.pkl'

with open(filepath, 'rb') as f:
	word_data = pickle.load(f)

# dict_keys(['total_word_counts', 'words', 'per_doc_word_counts'])

word_entropy = dict()
for word in word_data['words']:
	word_entropy[word] = calc_word_entropy(word_data['per_doc_word_counts'][word], word_data['total_word_counts'][word])
word_entropy = [(k, word_entropy[k], word_data['total_word_counts'][k]) for k in word_data['words']]
word_entropy = sorted(word_entropy, key=lambda x: x[1], reverse=True)
max_word_len = max([len(word) for word in word_data['words']])

for i in range(len(word_entropy)):
	print('{}{} : {:.2f}, {}'.format(word_entropy[i][0], ' ' * (max_word_len + 1 - len(word_entropy[i][0])), word_entropy[i][1], word_entropy[i][2]))


