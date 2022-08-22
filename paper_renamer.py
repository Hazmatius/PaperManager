import pdfplumber
import decimal
import os


def get_filtered_text(file_to_parse):
	all_words = list()
	with pdfplumber.open(file_to_parse) as pdf:
		max_page_number = min(len(pdf.pages), 3)
		for i in range(max_page_number):
			text = pdf.pages[i]
			words = text.extract_words(extra_attrs=['size'])
			for word in words:
				for key in word.keys():
					if isinstance(word[key], decimal.Decimal):
						word[key] = float(word[key])
			all_words.extend(words)
		return all_words


def title_from_words(all_words):
	font_sizes = dict()
	for word in all_words:
		if word['size'] in font_sizes.keys():
			font_sizes[word['size']].append(word['text'])
		else:
			font_sizes[word['size']] = [word['text']]
	sizes = sorted(list(font_sizes.keys()), reverse=True)
	title_words = font_sizes[sizes[0]]
	title = ' '.join(title_words)
	return title


def find_title(file_to_parse):
	return title_from_words(get_filtered_text(file_to_parse))


folder = '/Users/raymondbaranski/Literature/target_folder'
items = [item for item in os.listdir(folder) if item.lower().endswith('.pdf')]
items = sorted(items)
n_items = len(items)
i = 0
for item in items:
	i += 1
	print('\r{} / {}\t\t\t'.format(i, n_items), end='')
	old_filepath = os.path.join(folder, item)
	title = find_title(old_filepath).replace(':', '--')
	new_filepath = os.path.join(folder, '{}.pdf'.format(title))
	if len(title) < 255 and not os.path.exists(new_filepath):
		try:
			os.rename(old_filepath, new_filepath)
		except:
			print('Error with "{}"\n'.format(title))
		# print(title)


# words = get_filtered_text('/Users/raymondbaranski/Desktop/fpsyg.2017.01166.pdf')
# title = find_title(words)
# print(title)

# print(dir(clean_text))
# print(clean_text)