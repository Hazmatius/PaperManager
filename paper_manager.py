import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter import simpledialog
from tkinter.messagebox import askyesno
from PIL import Image, ImageTk
from mendeley import Mendeley
import requests, re
import sys, subprocess
import urllib.request
from urllib.error import HTTPError
import ssl
import os
import re
from pdf2image import convert_from_path
import pdfplumber
import textract
import matplotlib.pyplot as plt
import numpy as np
import time
from collections import defaultdict
import json
import pickle
import Levenshtein as lev
from tkinter import PhotoImage
from gscholar import query
from pathvalidate import sanitize_filename
from multiprocessing import Process, Queue
from crossref.restful import Works
import webbrowser
import textwrap


NO_STATUS = 'no_status'  # we haven't requested records yet [gray circle]
CONFIRMED = 'confirmed'  # user has confirmed bibtex [blue circle]
MATCH = 'match'  # there is a match with good bibtex [green circle]
NEAR_MATCH = 'near_match'  # there are near-matches with good bibtex [green exclamation point]
BAD_BIBTEX = 'bad_bibtex'  # there are matches or near-matches with bad or NO bibtex [yellow exclamation point]
BAD_RECORDS = 'bad_records'  # there are records, but there isn't a good match [red circle]
NO_RECORDS = 'no_records'  # there are simply no records at all [red exclamation point]
STATUS_OPTS = [NO_STATUS, CONFIRMED, MATCH, NEAR_MATCH, BAD_BIBTEX, BAD_RECORDS, NO_RECORDS]

MENDELEY = 'mendeley'
GOOGLE = 'google'
CROSSREF = 'crossref'

ALL_DOCUMENTS = 'All Documents'
SELECTED_DOCUMENTS = 'Selected Documents'
BIBTEX_FILTER = 'Bibtex Filter'
TAG_FILTER = 'Tag Filter'

PDFPLUMBER = 'pdfplumber'
TEXTRACT = 'textract'


def write_to_clipboard(output):
	process = subprocess.Popen(
		'pbcopy', env={'LANG': 'en_US.UTF-8'}, stdin=subprocess.PIPE)
	process.communicate(output.encode('utf-8'))


def organize_raw_text(raw_text, line_width):
	return textwrap.fill(raw_text, line_width)


def parse_word_search(word_search, srch):
	# we assume that there could be negation '~', grouping '()', and '*', and or '+'.
	# ~(neuron*apple)+baller
	# the problem is we have nested statements, so we can't really just use string split
	# we need to write... an actual parser! Oh no!
	# wait... can't we just, like, us eval to create a conditional? This could be TERRIBLE though.
	# basically, what we would do is this:
	word_search = word_search.lower()
	terms = re.findall(r'[^~()\+\*]*', word_search)
	terms = list(set([term for term in terms if term != '']))
	redict = dict()
	for term in terms:
		redict[term] = '(\'{}\' in {})'.format(term, srch)
	for term in terms:
		word_search = re.sub(r'\b'+term+r'\b', redict[term], word_search)
		# word_search = word_search.replace(term, redict[term])
	word_search = word_search.replace('+', ' or ').replace('*', ' and ').replace('~', 'not ')

	def func(doc):
		try:
			result = eval(word_search)
			return result
		except Exception as e:
			print(e)
			return False

	return func


def rgbtohex(color):
	r = int(color[0] * 255)
	g = int(color[1] * 255)
	b = int(color[2] * 255)
	return f'#{r:02x}{g:02x}{b:02x}'


def get_pdfs_in_folder(folder):
	files = os.listdir(folder)
	full_paths = [os.path.join(folder, file) for file in files]
	pdfs = [path for path in full_paths if '.pdf' in path.lower()]
	subdirs = [path for path in full_paths if os.path.isdir(path)]
	if len(subdirs) > 0:
		for subdir in subdirs:
			pdfs.extend(get_pdfs_in_folder(subdir))
	return pdfs


def is_good_char(e):
	return e.isalnum() or e == ' '


def clean_string(a_string):
	return ''.join(e for e in a_string.lower() if is_good_char(e))


def title_from_bibtex(bib_dict):
	title = bib_dict['title']
	title = title.replace(r'\textquotedblleft', '"')
	title = title.replace(r'\textquotedblright', '"')
	title = title.replace(r'\textendash', '-')
	title = title.replace(r'\textquotesingle', '\'')
	return title


def get_similarity_score(res_title, doc_title):
	"""
	Calculates the Levenshtein ratio between two titles
	:param res_title:
	:param doc_title:
	:return:
	"""
	res_title = clean_string(res_title)
	cur_title = clean_string(doc_title)
	similarity = lev.ratio(res_title, cur_title)
	return similarity


def bibtex_quality_score(bibtex):
	return 1


def bibtex_to_dict(bibtex_string):
	bibtex_fields = [bibtex_field for bibtex_field in bibtex_string.split('\n') if '=' in bibtex_field]
	bibdict = dict()
	for field in bibtex_fields:
		key, value = field.split('=')
		key = key.replace(' ', '').replace('\t', '')
		value = value[1:-1].replace('{', '').replace('}', '')
		bibdict[key] = value
	return bibdict


def sort_keys(keys, keysort, exclude):
	sorted_keys = list()
	for key in keysort:
		if key in keys:
			sorted_keys.append(key)
			keys.remove(key)
	for key in exclude:
		if key in keys:
			keys.remove(key)
	sorted_keys.extend(keys)
	return sorted_keys


def calc_word_entropy(doc_word_counts, total_word_count):
	"""
	This function should return 1 for a perfectly spread-out distribution, and, I dunno... 0 if in only one document?
	"""
	s_vals = doc_word_counts / total_word_count
	entropy = -np.sum(s_vals * np.log2(s_vals)) / np.log2(len(doc_word_counts))
	return entropy


class Document(object):
	def __init__(self, filepath, doc_id):
		self.filepath = filepath
		self.filename = os.path.basename(self.filepath)
		self.folder = os.path.dirname(self.filepath)
		self.title = self.filename
		self.title = self.title.replace('.PDF', '')
		self.title = self.title.replace('.pdf', '')
		self.document_id = doc_id
		self.records = list()
		self.tags = set()

		self.infolist = list()

		self.bibtex = None
		self.raw_text = None
		self.word_counts = dict()
		self.words = set()
		self.keywords = set()

		self.pdf_ids = list()
		self.photos = list()
		self.x_tolerance = 1
		self.notes = None

		self.saved = False
		self.bibtex_status = 'no_status'

		self.TITLE_MATCH = 1
		self.TITLE_NEAR_MATCH = 0.9

	def __getstate__(self):
		state = self.__dict__.copy()
		state['photos'] = list()
		state['tags'] = [tag.tagword for tag in self.tags]
		return state

	def _tag(self, tag):
		self.tags.add(tag)
		self.saved = False

	def _untag(self, tag):
		self.tags.remove(tag)
		self.saved = False

	def get_tags_string(self):
		return ', '.join([tag.tagword for tag in self.tags])

	def calc_safe_filename(self):
		return sanitize_filename(self.title.replace(':', '-').replace('.', '-')) + '.pdf'

	def check_if_filename_available(self, filename):
		return not os.path.exists(os.path.join(self.folder, filename))

	def rename_file(self, new_filename):
		new_filepath = os.path.join(self.folder, new_filename)
		os.rename(self.filepath, new_filepath)
		self.filepath = new_filepath
		self.filename = new_filename

	def load_first_pages(self):
		factor = 550 / 800
		if len(self.photos) < 2:
			pages = convert_from_path(self.filepath, size=(800 * factor, 1000 * factor), last_page=2)
			for i in range(len(pages)):
				self.photos.append(ImageTk.PhotoImage(pages[i]))

	def load_pdf(self, ui):
		ui.pdf.delete('1.0', END)

		self.load_first_pages()

		for photo in self.photos:
			self.pdf_ids.append(ui.pdf.image_create(END, image=photo))
			ui.pdf.insert(END, '\n\n')

	def calculate_keywords(self, dictionary):
		for word in self.words:
			if word in dictionary.keywords:
				self.keywords.add(word)
		rm_words = set()
		for word in self.keywords:
			if word in dictionary.trashwords or word in dictionary.normwords:
				rm_words.add(word)
		self.keywords = self.keywords - rm_words

	def load_text(self, method):
		if self.raw_text is None:
			self.saved = False
			text = ''
			if method == PDFPLUMBER:
				with pdfplumber.open(self.filepath) as pdf_file:
					for page in pdf_file.pages:
						page_text = page.extract_text(x_tolerance=self.x_tolerance)
						if page_text is not None:
							text += page_text
			elif method == TEXTRACT:
				text = textract.process(self.filepath).decode('utf-8')
			dictionary = defaultdict(lambda: 0)
			text = re.sub('[\n()\[\]{}0-9&@…#%]', '', text)
			text = re.sub('[.,;:?…“”\'‘’/\"]', ' ', text)
			text = ' '.join(text.split())
			self.raw_text = text

			words = text.split(' ')
			words = [word.lower() for word in words if len(word) >= 3]
			for word in words:
				dictionary[word] += 1
			self.word_counts = dict(dictionary)
			self.words = set(dictionary.keys())

	def update_notes(self, directory, ui):
		new_text = ui.notes.get('1.0', END).rstrip('\n')
		if new_text != self.notes:
			self.notes = new_text
			self.save(directory)

	def save(self, directory):
		with open(os.path.join(directory, str(self.document_id) + '.docinf'), 'wb') as f:
			pickle.dump(self, f)
		self.saved = True

	def calculate_document_status(self):
		# look through all records
		sim_scores = np.zeros(len(self.records))
		has_bibtex = np.zeros(len(self.records))
		bibtxscore = np.zeros(len(self.records))
		for i in range(len(self.records)):
			sim_scores[i] = self.records[i]['similarity']
			has_bibtex[i] = 'bibtex' in self.records[i]
			if 'bibtex' in self.records:
				bibtxscore[i] = bibtex_quality_score(self.records[i]['bibtex'])

		match = np.logical_and(sim_scores == self.TITLE_MATCH, has_bibtex == 1, bibtxscore == 1)
		if np.any(match):
			idxs = list(np.where(match)[0])
			self.bibtex = self.records[idxs[0]]['bibtex']
			self.bibtex_status = MATCH
			return self.bibtex_status
		near_match = np.logical_and(sim_scores > self.TITLE_NEAR_MATCH, has_bibtex == 1, bibtxscore == 1)
		if np.any(near_match):
			self.bibtex_status = NEAR_MATCH
			return self.bibtex_status
		bad_bibtex = sim_scores > self.TITLE_NEAR_MATCH
		if np.any(bad_bibtex):
			self.bibtex_status = BAD_BIBTEX
			return self.bibtex_status
		if len(self.records) > 0:
			self.bibtex_status = BAD_RECORDS
			return self.bibtex_status
		if len(self.records) == 0:
			self.bibtex_status = NO_RECORDS
			return self.bibtex_status
		raise RuntimeError('This line should literally never run, what have you done you fool.')

	def extend_records(self, results):
		for res in results:
			if res not in self.records:
				self.records.append(res)


class NetScraper(object):
	def __init__(self, refmanager):
		self.refmanager = refmanager
		ssl._create_default_https_context = ssl._create_unverified_context
		self.BASE_URL = 'http://dx.doi.org/'
		client_id = 9217
		client_secret = 'hMTvq5N2z29ZDzKp'
		'https://localhost:5000/oauth'

		redirect_uri = 'https://127.0.0.1:5000/oauth'
		mendeley = Mendeley(client_id, client_secret)
		auth = mendeley.start_implicit_grant_flow()
		login_url = auth.get_login_url()

		res = requests.post(login_url, allow_redirects=False, data={
			'username': 'alexbaranski@gmail.com',
			'password': 'the great library of alexander'
		})
		auth_response = res.headers['Location']
		# print(auth_response)
		self.session = auth.authenticate(auth_response.replace('http', 'https'))

		self.works = Works()

		# self.values = {
		# 	ALL_DOCUMENTS: 0,
		# 	SELECTED_DOCUMENTS: 1,
		# 	BIBTEX_FILTER: 2,
		# 	TAG_FILTER: 3
		# }
		self.values = {
			ALL_DOCUMENTS: 0,
			SELECTED_DOCUMENTS: 1
		}
		self.num_to_val = {self.values[key]: key for key in self.values.keys()}

	def get_bibtex(self, doi):
		"""
		Returns bibtex from doi, used when we get a doi from Mendeley
		:param doi:
		:return:
		"""
		url = self.BASE_URL + doi
		req = urllib.request.Request(url)
		req.add_header('Accept', 'application/x-bibtex')
		try:
			with urllib.request.urlopen(req) as f:
				bibtex = f.read().decode()
			return bibtex
		except HTTPError as e:
			if e.code == 404:
				print('DOI not found.')
			else:
				print('Service unavailable.')
			return None

	def mendeley_search(self, title):
		"""
		Queries Mendeley by title, returns a list of dictionaries, possibly including bibtex
		:param title:
		:return:
		"""
		net_results = self.session.catalog.search(title)
		results = list()
		max_iter = 10
		index = 0
		for doc in net_results.iter():
			index += 1
			result = self._get_mendeley_record(doc, title)
			results.append(result)
			if index >= max_iter:
				break
		return results

	def google_search(self, title):
		"""
		Scrapes Google Scholar by searching by title, returns list of dictionaries that include bibtex
		:param title:
		:return:
		"""
		net_results = query(title)
		results = list()
		for doc in net_results:
			result = self._get_google_record(doc, title)
			results.append(result)
		return results

	def crossref_search(self, title):
		net_results = self.works.query(bibliographic=title)
		results = list()
		max_iter = 10
		index = 0
		for doc in net_results:
			index += 1
			result = self._get_crossref_record(doc, title)
			results.append(result)
			if index >= max_iter:
				break
		return results

	def _get_google_record(self, result, doc_title):
		"""
		Prepares a dictionary of results from a Google Scholar result
		:param result:
		:param doc_title:
		:return:
		"""
		results_dict = bibtex_to_dict(result)
		similarity = get_similarity_score(results_dict['title'], doc_title)
		results_dict['similarity'] = similarity
		results_dict['bibtex'] = result
		return results_dict

	def _get_mendeley_record(self, result, doc_title):
		"""
		Prepares a dictionary of results from a Mendeley result
		:param result:
		:param doc_title:
		:return:
		"""
		field_names = ['title', 'authors', 'identifiers', 'keywords', 'link', 'source', 'year']
		max_field_len = max([len(field_name) for field_name in field_names])
		results_dict = dict()
		results_string = ''

		# set to lower case and remove special characters
		similarity = get_similarity_score(result.title, doc_title)

		for i in range(len(field_names)):
			field_name = field_names[i]
			field_value = getattr(result, field_name)
			if field_name == 'authors':
				field_value = self.get_authors_pretty(field_value)
			padding = (max_field_len - len(field_name)) * ' '
			results_dict[field_name] = field_value
			if field_name == 'title':
				results_string += '{}{} : {} ({:.2f}% match)'.format(padding, field_name, field_value, similarity*100)
			else:
				results_string += '{}{} : {}'.format(padding, field_name, field_value)
			if i < len(field_names) - 1:
				results_string += '\n'
		results_dict['similarity'] = similarity
		if similarity > 0.9:
			if result.identifiers is not None and 'doi' in result.identifiers:
				doi = result.identifiers['doi']
				bibtex = self.get_bibtex(doi)
				if bibtex is not None:
					results_dict['bibtex'] = bibtex
		return results_dict

	def _get_crossref_record(self, result, doc_title):
		keys = list(result.keys())
		for key in keys:
			result[key.lower()] = result.pop(key)
		field_names = list(result.keys())
		# field_names = ['title', 'author', 'publisher', 'published', 'doi']
		max_field_len = max([len(field_name) for field_name in field_names])
		results_dict = dict()
		results_string = ''

		if 'title' in result.keys():
			if len(result['title']) == 1:
				result['title'] = result['title'][0]
			elif len(result['title']) == 0:
				result['title'] = 'no title'
			else:
				sims = [get_similarity_score(title, doc_title) for title in result['title']]
				max_sim_idx = np.argmax(sims)
				result['title'] = result['title'][max_sim_idx]

		similarity = get_similarity_score(result['title'], doc_title)

		for i in range(len(field_names)):
			field_name = field_names[i]
			field_value = result[field_name]
			# field_value = getattr(result, field_name)
			if field_name == 'authors' or field_name == 'author':
				field_value = self.get_authors_pretty_crossref(field_value)
			padding = (max_field_len - len(field_name)) * ' '
			results_dict[field_name] = field_value
			if field_name == 'title':
				results_string += '{}{} : {} ({:.2f}% match)'.format(padding, field_name, field_value, similarity * 100)
			else:
				results_string += '{}{} : {}'.format(padding, field_name, field_value)
			if i < len(field_names) - 1:
				results_string += '\n'
		results_dict['similarity'] = similarity
		if similarity > 0.9:
			if 'doi' in result.keys():
				doi = result['doi']
				bibtex = self.get_bibtex(doi)
				if bibtex is not None:
					results_dict['bibtex'] = bibtex
		return results_dict

	def get_authors_pretty(self, author_list):
		"""
		Converts Mendeley author lists into strings
		:param author_list:
		:return:
		"""
		if isinstance(author_list, list):
			return ['{} {}'.format(author.first_name, author.last_name) for author in author_list]
		else:
			if author_list is not None:
				return '{} {}'.format(author_list.first_name, author_list.last_name)
			else:
				return None

	def get_authors_pretty_crossref(self, author_list):
		try:
			if isinstance(author_list, list):
				return ['{} {}'.format(author['given'], author['family']) for author in author_list]
			else:
				if author_list is not None:
					return '{} {}'.format(author_list['given'], author_list['family'])
				else:
					return None
		except:
			return author_list

	def bulk_scrape_interface(self):
		self.window = tk.Toplevel()
		self.window.wm_title('Bulk Netscrape')

		self.v = IntVar(self.window, 0)

		buttons_frame = Frame(self.window)
		buttons_frame.pack(side=LEFT)

		# Dictionary to create multiple buttons

		# Loop is used to create multiple Radiobuttons
		# rather than creating each button separately
		for (text, value) in self.values.items():
			f = Frame(buttons_frame, height=25, width=200)
			f.pack(side=TOP)
			f.pack_propagate(0)
			Radiobutton(f, text=text, variable=self.v, value=value).pack(side=LEFT, ipady=5)

		self.options_text = Text(self.window, relief=GROOVE, borderwidth=2, width=25, height=6)
		self.options_text.pack(side=LEFT)

		search_frame = Frame(self.window)
		search_frame.pack(side=LEFT)

		bulk_mendeley_button = Button(search_frame, text='Mendeley', command=self.bulk_check_mendeley)
		bulk_mendeley_button.pack(side=TOP)
		bulk_crossref_button = Button(search_frame, text='Crossref', command=self.bulk_check_crossref)
		bulk_crossref_button.pack(side=TOP)

	def get_bulk_documents(self):
		opt = self.num_to_val[self.v.get()]
		if opt == ALL_DOCUMENTS:
			return self.refmanager.documents.values()
		elif opt == SELECTED_DOCUMENTS:
			return self.refmanager.get_selected_documents()
		else:
			print('Invalid opt "{}"'.format(opt))
			return None

	def bulk_check_documents(self, method):
		bulk_docs = self.get_bulk_documents()
		documents_to_check = list()
		for doc in bulk_docs:
			if doc.bibtex_status not in [CONFIRMED, MATCH, NEAR_MATCH]:
				documents_to_check.append(doc)

		# progress_var = DoubleVar()
		progress_bar_window = tk.Toplevel()
		progress_bar_window.wm_title('Checking {}...'.format(method.capitalize()))
		top_frame = Frame(progress_bar_window)
		top_frame.pack(side=TOP)

		progressbar = ttk.Progressbar(
			top_frame,
			value=0,
			orient='horizontal',
			mode='determinate',
			length=500,
			maximum=len(documents_to_check)
		)
		progressbar.pack(side=LEFT)
		doc_var = StringVar()
		prog_var = StringVar()

		progress_bar_label = Label(top_frame, height=2, width=10, textvariable=prog_var)
		progress_bar_label.pack(side=LEFT)
		doc_title_label = Label(progress_bar_window, height=4, width=100, textvariable=doc_var)
		doc_title_label.pack(side=TOP)

		progress_bar_window.update()
		progress_bar_window.update_idletasks()

		for i in range(len(documents_to_check)):
			doc = documents_to_check[i]
			progressbar.step(1)
			prog_var.set('{}/{}'.format(i + 1, len(documents_to_check)))
			doc_var.set(doc.title)
			time.sleep(0.02)
			self.refmanager.update_records(doc, method)
			doc.save(self.refmanager.documents_directory)
			# self.ui.root.update_idletasks()
			progress_bar_window.update()
			progress_bar_window.update_idletasks()
		progress_bar_window.destroy()
		self.refmanager.display_docs(bulk_docs)

	def bulk_check_mendeley(self):
		self.bulk_check_documents(MENDELEY)

	def bulk_check_crossref(self):
		self.bulk_check_documents(CROSSREF)


class Dictionary(object):
	def __init__(self, refmanager):
		self.refmanager = refmanager
		self.all_words = defaultdict(lambda: 0)
		self.keywords = list()
		self.normwords = list()
		self.trashwords = list()
		self.unsorted = list()

	def get_full_word_information(self):
		per_doc_word_counts = dict()
		words = list()
		for doc in self.refmanager.documents.values():
			doc.word_counts = defaultdict(lambda: 0, doc.word_counts)
		for word in self.unsorted:
			print('\r{}'.format(word) + '\t'*5, end='')
			if self.all_words[word] > 10:
				words.append(word)
				per_doc_word_counts[word] = [doc.word_counts[word] for doc in self.refmanager.documents.values()]
		print('')
		data_dump = {
			'total_word_counts': self.all_words,
			'words': words,
			'per_doc_word_counts': per_doc_word_counts
		}
		with open('/Users/raymondbaranski/Desktop/some_word_data/word_data.pkl', 'wb') as f:
			pickle.dump(data_dump, f)
		exit()

	def calc_word_entropy(self, word):
		all_docs = list(self.refmanager.documents.values())
		doc_word_counts = np.zeros(len(all_docs))
		total_word_count = self.all_words[word]
		for i in range(len(all_docs)):
			doc_word_counts[i] = all_docs[i].word_counts[word]
		return calc_word_entropy(doc_word_counts, total_word_count)

	def _dictionary_keydown(self, this_particular_listbox, this_particular_word_list, e):
		if e.char == 'a' or e.char == 's' or e.char == 'd':
			selected_item = this_particular_listbox.selection()
			children = this_particular_listbox.get_children()
			item_index = children.index(selected_item[0])
			word = this_particular_listbox.item(selected_item)['text']

			if word in this_particular_word_list:
				this_particular_word_list.remove(word)
				this_particular_listbox.delete(selected_item)
				children = this_particular_listbox.get_children()
				if children:
					try:
						this_particular_listbox.focus(children[item_index])
						this_particular_listbox.selection_set(children[item_index])
					except:
						try:
							this_particular_listbox.focus(children[item_index - 1])
							this_particular_listbox.selection_set(children[item_index - 1])
						except:
							pass
				if e.char == 'a':
					self.keywords.append(word)
					self.keywords_listbox.insert('', 0, text=word, values=(word, self.all_words[word]))
				elif e.char == 's':
					self.normwords.append(word)
					self.normwords_listbox.insert('', 0, text=word, values=(word, self.all_words[word]))
				elif e.char == 'd':
					self.trashwords.append(word)
					self.trashwords_listbox.insert('', 0, text=word, values=(word, self.all_words[word]))

		self.refmanager.save_metadata()

	def unsorted_words_listbox_keydown(self, e):
		this_particular_listbox = self.unsorted_words_listbox
		this_particular_word_list = self.unsorted
		self._dictionary_keydown(this_particular_listbox, this_particular_word_list, e)

	def keywords_listbox_keydown(self, e):
		this_particular_listbox = self.keywords_listbox
		this_particular_word_list = self.keywords
		self._dictionary_keydown(this_particular_listbox, this_particular_word_list, e)

	def normwords_listbox_keydown(self, e):
		this_particular_listbox = self.normwords_listbox
		this_particular_word_list = self.normwords
		self._dictionary_keydown(this_particular_listbox, this_particular_word_list, e)

	def trashwords_listbox_keydown(self, e):
		this_particular_listbox = self.trashwords_listbox
		this_particular_word_list = self.trashwords
		self._dictionary_keydown(this_particular_listbox, this_particular_word_list, e)

	def create_word_listbox(self, parent, height, side):
		subframe = Frame(parent)
		subframe.pack(side=side, expand=TRUE)

		new_listbox = ttk.Treeview(subframe, column=('w1', 'w2'), show='headings', height=height)
		new_listbox.column('# 1', anchor=CENTER, stretch=NO, minwidth=250, width=250)
		new_listbox.heading('# 1', text='Word')
		new_listbox.column('# 2', anchor=CENTER, stretch=NO, minwidth=100, width=100)
		new_listbox.heading('# 2', text='Count')
		new_listbox.pack(in_=subframe, side=LEFT, expand=TRUE)

		new_scrollbar = Scrollbar(subframe)
		new_scrollbar.pack(side=LEFT, fill=BOTH)
		new_listbox.config(yscrollcommand=new_scrollbar.set)
		new_scrollbar.config(command=new_listbox.yview)

		return new_listbox, new_scrollbar

	def open_dictionary_window(self):
		window = tk.Toplevel()
		window.wm_title('Dictionary')

		treeview_frame_top = Frame(window)
		treeview_frame_top.pack(side=TOP, fill=BOTH, expand=TRUE)

		button_frame_bottom = Frame(window)
		button_frame_bottom.pack(side=TOP, fill=BOTH, expand=TRUE)
		load_text_button = Button(button_frame_bottom, text='Load Text', command=self.parallel_load_text)
		load_text_button.pack(side=LEFT)
		# calc_dictionary_button = Button(button_frame_bottom, text='Calc. Dictionary', command=self.parallel_calculate_dictionary)
		# calc_dictionary_button.pack(side=LEFT)

		dictionary_frame_left = Frame(treeview_frame_top)
		dictionary_frame_left.pack(side=LEFT, fill=BOTH, expand=TRUE)

		self.unsorted_words_listbox, self.unsorted_scrollbar = self.create_word_listbox(dictionary_frame_left, 32, LEFT)
		self.unsorted_words_listbox.bind("<KeyPress>", self.unsorted_words_listbox_keydown)

		dictionary_frame_right = Frame(treeview_frame_top)
		dictionary_frame_right.pack(side=LEFT, fill=BOTH, expand=TRUE)

		self.keywords_listbox, self.keywords_scrollbar = self.create_word_listbox(dictionary_frame_right, 9, TOP)
		self.keywords_listbox.bind("<KeyPress>", self.keywords_listbox_keydown)

		self.normwords_listbox, self.normwords_scrollbar = self.create_word_listbox(dictionary_frame_right, 9, TOP)
		self.normwords_listbox.bind("<KeyPress>", self.normwords_listbox_keydown)

		self.trashwords_listbox, self.trashwords_scrollbar = self.create_word_listbox(dictionary_frame_right, 9, TOP)
		self.trashwords_listbox.bind("<KeyPress>", self.trashwords_listbox_keydown)

		self.update_word_lists()

	def update_word_lists(self):
		self.update_word_list(self.unsorted_words_listbox, self.unsorted)
		self.update_word_list(self.keywords_listbox, self.keywords)
		self.update_word_list(self.normwords_listbox, self.normwords)
		self.update_word_list(self.trashwords_listbox, self.trashwords)

	def update_word_list(self, listbox, word_list, **kwargs):
		if 'entropy' in kwargs and kwargs['entropy']:
			counts = [self.calc_word_entropy(word) for word in word_list]
		else:
			counts = [self.all_words[word] for word in word_list]
		sort_idxs = np.flip(np.argsort(counts))
		words = [word_list[i] for i in sort_idxs]
		counts = [counts[i] for i in sort_idxs]

		listbox.delete(*listbox.get_children())
		for i in range(len(words)):
			# print(words[i], counts[i])
			listbox.insert('', END, text=words[i], values=(words[i], counts[i]))

	@staticmethod
	def load_doc_text(queue):
		while True:
			msg = queue.get()
			if msg == 'done':
				print('Finished')
				break
			else:
				document, folder, index, maximum = msg['document'], msg['folder'], msg['index'], msg['maximum']
				try:
					document.load_text(TEXTRACT)
					document.save(folder)
					print('{}/{}'.format(index, maximum))
				except Exception as e:
					print('Failure on loading text for [{}][{}]'.format(document.title, document.filepath))
					print(e)

	def parallel_load_text(self):
		n_workers = 3
		self.all_words = defaultdict(lambda: 0)

		if __name__ == '__main__':
			queue = Queue()

			workers = list()
			for i in range(n_workers):
				worker = Process(target=self.load_doc_text, args=((queue),))
				# worker.daemon = True
				worker.start()
				workers.append(worker)

			index = 1
			maximum = len(self.refmanager.documents.values())
			for doc in self.refmanager.documents.values():
				msg = {'document': doc, 'folder': self.refmanager.documents_directory, 'index': index, 'maximum': maximum}
				index += 1
				queue.put(msg)
			for i in range(n_workers):
				msg = 'done'
				queue.put(msg)

			for worker in workers:
				worker.join()

			self.refmanager.reload_files()

	def parallel_calculate_dictionary(self):
		self.parallel_load_text()
		self.compile_dictionary()
		print('\nCompiled dictionary.')

	def calculate_dictionary(self):
		self.all_words = defaultdict(lambda: 0)
		counter = 0
		maximum = len(self.refmanager.documents.values())
		for doc in self.refmanager.documents.values():
			counter += 1
			try:
				print('\rCalculating {}/{}...'.format(counter, maximum) + '\t'*5, end='')
				if len(doc.word_counts) == 0:
					doc.load_text(TEXTRACT)
			except Exception as e:
				print('A failure in loading [{}][{}].'.format(doc.title, doc.filepath))
				print(e)
		print('')

		self.compile_dictionary()

	def compile_dictionary(self):
		for doc in self.refmanager.documents.values():
			for word in doc.word_counts.keys():
				self.all_words[word] += doc.word_counts[word]

		for word in self.all_words.keys():
			if not (word in self.keywords or word in self.normwords or word in self.trashwords or word in self.unsorted):
				if self.all_words[word] == 1:
					self.trashwords.append(word)
				else:
					self.unsorted.append(word)
		self.update_word_lists()

	def load_data(self, dictionary_data):
		self.all_words = defaultdict(int, dictionary_data['all_words'])
		self.keywords = dictionary_data['keywords']
		self.normwords = dictionary_data['normwords']
		self.trashwords = dictionary_data['trashwords']
		self.unsorted = dictionary_data['unsorted']

	def get_data(self):
		dictionary_data = {
			'all_words': dict(self.all_words),
			'keywords': self.keywords,
			'normwords': self.normwords,
			'trashwords': self.trashwords,
			'unsorted': self.unsorted
		}
		return dictionary_data


class FolderWatcher(object):
	def __init__(self, refmanager):
		self.refmanager = refmanager
		self.watched_folders = set()

	def open_folder_window(self):
		window = tk.Toplevel()
		window.wm_title('Watched Folders')
		self.folder_listbox = Listbox(window, width=50, height=10)
		self.folder_listbox.pack()
		# scrollbar = Scrollbar(window)
		# scrollbar.pack(side=RIGHT, fill=BOTH)
		self.update_folder_list()
		add_folder_button = Button(window, text='Add Folder', command=self.add_folder)
		add_folder_button.pack()
		remove_folder_button = Button(window, text='Remove Folder', command=self.remove_folder)
		remove_folder_button.pack()

	def update_folder_list(self):
		self.folder_listbox.delete(0, END)
		for folder in self.watched_folders:
			self.folder_listbox.insert(END, folder)

	def remove_folder(self):
		try:
			folder = self.folder_listbox.get(self.folder_listbox.curselection())
			if folder != '':
				self.watched_folders.remove(folder)
				rm_docs = list()
				for key, doc in self.refmanager.documents.items():
					if doc.folder == folder:
						rm_docs.append(key)
				for key in rm_docs:
					del self.refmanager.documents[key]
			self.update_folder_list()
			self.refmanager.display_docs(self.refmanager.documents.values())
		except:
			pass

	def add_folder(self):
		foldername = filedialog.askdirectory(initialdir='/Users/raymondbaranski/Literature')
		if foldername != '':
			if foldername not in self.watched_folders:
				self.watched_folders.add(foldername)
				pdfs = get_pdfs_in_folder(foldername)
				for pdf in pdfs:
					if pdf not in self.refmanager.documents.keys():
						new_doc = Document(pdf, self.refmanager.doc_id_counter)
						self.refmanager.doc_id_counter += 1
						self.refmanager.documents[pdf] = new_doc
						self.refmanager.id_to_doc[new_doc.document_id] = new_doc
		self.refmanager.display_docs(self.refmanager.documents.values())
		self.update_folder_list()

	def check_for_new_documents(self):
		# compose file_list
		for folder in self.watched_folders:
			# print('Looking in {}...'.format(folder))
			pdfs = get_pdfs_in_folder(folder)
			for pdf in pdfs:
				if pdf not in self.refmanager.documents.keys():
					print('found new document {}'.format(pdf))
					new_doc = Document(pdf, self.refmanager.doc_id_counter)
					self.refmanager.doc_id_counter += 1
					self.refmanager.documents[pdf] = new_doc
					self.refmanager.id_to_doc[new_doc.document_id] = new_doc
		self.refmanager.display_docs(self.refmanager.documents.values())


class Tag(object):
	def __init__(self, tagword):
		self.tagword = tagword
		self.document_ids = set()

	def set_tagword(self, tagword):
		self.tagword = tagword

	def _add_document(self, document):
		self.document_ids.add(document.document_id)

	def _remove_document(self, document):
		self.document_ids.remove(document.document_id)


class TagManager(object):
	def __init__(self, refmanager):
		self.refmanager = refmanager
		self.tags = dict()

	def load_tag_data(self, tags):
		self.tags = tags

	def open_tag_window(self):
		self.window = tk.Toplevel()
		self.window.wm_title('Tags')
		self.tag_listbox = ttk.Treeview(self.window, column=('c1'), height=10, show=['headings'])
		# self.tag_listbox = Listbox(self.window, selectmode = 'multiple', width=50, height=10)
		self.tag_listbox.pack()
		self.update_tag_list()

		button_frame1 = Frame(self.window)
		button_frame1.pack(side=BOTTOM, fill=BOTH, expand=TRUE)
		self.add_tag_button = Button(button_frame1, text='Add', command=self.add_tag)
		self.add_tag_button.pack(side=LEFT)

		self.delete_tag_button = Button(button_frame1, text='Delete', command=self.delete_tag)
		self.delete_tag_button.pack(side=LEFT)

		self.modify_tag_button = Button(button_frame1, text='Modify', command=self.modify_tag)
		self.modify_tag_button.pack(side=LEFT)

		button_frame2 = Frame(self.window)
		button_frame2.pack(side=BOTTOM, fill=BOTH, expand=TRUE)

		self.merge_copy_button = Button(button_frame2, text='Merge Copy', command=self.merge_copy_tags)
		self.merge_copy_button.pack(side=LEFT)

		self.tag_button = Button(button_frame2, text='Tag', command=self.tag_documents)
		self.tag_button.pack(side=LEFT)

		self.untag_button = Button(button_frame2, text='Untag', command=self.untag_documents)
		self.untag_button.pack(side=LEFT)

	def update_doc_view(self):
		self.refmanager.select_document(None)

	def get_selected_tags(self):
		selected_items = self.tag_listbox.selection()
		tags = [self.tag_listbox.item(i)['values'][0] for i in selected_items]
		return tags

	def update_tag_list(self):
		self.tag_listbox.delete(*self.tag_listbox.get_children())
		i = 0
		for tag in self.tags.keys():
			self.insert_tag(i, tag, END)
			i += 1
		self.tag_listbox.tag_configure('light', background='#aaaaaa')
		self.tag_listbox.tag_configure('gray', background='#cccccc')

	def insert_tag(self, i, tag, index):
		values = (tag,)
		if i % 2 == 0:
			ctag = 'light'
		else:
			ctag = 'gray'
		self.tag_listbox.insert('', index, values=values, tag=ctag)

	def tag_documents(self):
		documents = self.refmanager.get_selected_documents()
		tags = self.get_selected_tags()
		for tag in tags:
			for doc in documents:
				self._tag_document(doc, self.tags[tag])
		self.update_doc_view()

	def untag_documents(self):
		documents = self.refmanager.get_selected_documents()
		tags = self.get_selected_tags()
		for tag in tags:
			for doc in documents:
				self._untag_document(doc, self.tags[tag])
		self.update_doc_view()

	def _tag_document(self, document, tag):
		tag._add_document(document)
		document._tag(tag)

	def _untag_document(self, document, tag):
		tag._remove_document(document)
		document._untag(tag)

	def add_tag(self):
		tagword = simpledialog.askstring('New tag', 'Tag')
		new_tag = self._add_tag(tagword)
		if new_tag is not None:
			self.update_tag_list()

	def _add_tag(self, tagword):
		if tagword in self.tags.keys():
			print('Tag "{}" already exists'.format(tagword))
			return None
		elif tagword is None or tagword == 'None':
			return None
		else:
			self.tags[tagword] = Tag(tagword)
			return self.tags[tagword]

	def delete_tag(self):
		tagwords = self.get_selected_tags()
		if len(tagwords) > 0:
			for tagword in tagwords:
				self._delete_tag(tagword)
			self.update_tag_list()
		self.update_doc_view()

	def _delete_tag(self, tagword):
		if tagword in self.tags.keys():
			tag = self.tags[tagword]
			for doc_id in tag.document_ids:
				document = self.refmanager.id_to_doc[doc_id]
				document._untag(tag)
			del self.tags[tagword]
		else:
			print('No such tag "{}"'.format(tagword))

	def modify_tag(self):
		tagwords = self.get_selected_tags()
		if len(tagwords) == 1:
			tagword = tagwords[0]
			tag = self.tags[tagword]
			for doc_id in tag.document_ids:
				self.refmanager.id_to_doc[doc_id].saved = False
			new_tagword = simpledialog.askstring('Modify tag', 'Tag')
			self._modify_tag(tagword, new_tagword)
			self.update_tag_list()
			self.update_doc_view()

	def _modify_tag(self, old_tagword, new_tagword):
		if old_tagword in self.tags.keys():
			tag = self.tags[old_tagword]
			tag.set_tagword(new_tagword)
			del self.tags[old_tagword]
			self.tags[new_tagword] = tag
		else:
			print('No such tag "{}"'.format(old_tagword))

	def _get_all_docs(self, tagwords):
		tags = list()
		for tagword in tagwords:
			if tagword in self.tags.keys():
				tags.append(self.tags[tagword])
			else:
				print('No such tag "{}"'.format(tagword))
				return None, None
		documents = [self.refmanager.id_to_doc[doc_id] for doc_id in set().union(*[tag.document_ids for tag in tags])]
		return tags, documents

	def merge_copy_tags(self):
		tagwords = self.get_selected_tags()
		if len(tagwords) > 0:
			new_tagword = simpledialog.askstring('New tag', 'Tag')
			self._merge_copy_tags(tagwords, new_tagword)
			self.update_tag_list()
			self.update_doc_view()

	def _merge_copy_tags(self, tagwords, new_tagword):
		tags, documents = self._get_all_docs(tagwords)
		if tags is not None and documents is not None:
			new_tag = self._add_tag(new_tagword)
			if new_tag is not None:
				for document in documents:
					self._tag_document(document, new_tag)
		else:
			return 1

	def _merge_delete_tags(self, tagwords, new_tagword):
		tags, documents = self._get_all_docs(tagwords)
		if tags is not None and documents is not None:
			new_tag = self._add_tag(new_tagword)
			if new_tag is not None:
				for document in documents:
					self._tag_document(document, new_tag)
				for tag in tags:
					self._delete_tag(tag)
		else:
			return 1


class ExportManager(object):
	def __init__(self, refmanager):
		self.refmanager = refmanager

	def open_export_interface(self):
		self.window = tk.Toplevel()
		self.window.wm_title('Export')

		self.export_bibtex_button = Button(self.window, text='Export Bibtex', command=self.export_bibtex)
		self.export_bibtex_button.pack(side=LEFT)

	def export_bibtex(self):
		documents = self.refmanager.get_selected_documents()
		bibtexs = list()
		invalid_bibtex = False
		for doc in documents:
			if doc.bibtex_status == MATCH or doc.bibtex_status == CONFIRMED:
				bibtexs.append(doc.bibtex)
			else:
				print('"{}" doesn\'t have proper bibtex'.format(doc.title))
				invalid_bibtex = True
		if not invalid_bibtex:
			filename = filedialog.asksaveasfilename(defaultextension='.bib')
			if not os.path.exists(filename) and filename != '':
				with open(filename, 'w') as f:
					f.write('\n'.join(bibtexs))
			else:
				print('"{}" already exists'.format(filename))


class RefManager(object):
	def __init__(self):
		self.netscraper = NetScraper(self)
		self.dictionary = Dictionary(self)
		self.folder_watcher = FolderWatcher(self)
		self.tag_manager = TagManager(self)
		self.export_manager = ExportManager(self)
		self.ui = UIManager(self)

		self.doc_id_counter = 0
		self.documents = dict()
		self.id_to_doc = dict()
		self.folder_listbox = None
		self.selected_document = None

		self.main_directory = '/Users/raymondbaranski/.paper_manager'
		if not os.path.exists(self.main_directory):
			os.mkdir(self.main_directory)
		self.documents_directory = os.path.join(self.main_directory, 'documents')
		if not os.path.exists(self.documents_directory):
			os.mkdir(self.documents_directory)
		self.trash_directory = os.path.join(self.main_directory, 'trash')
		if not os.path.exists(self.trash_directory):
			os.mkdir(self.trash_directory)

		resource_folder = '/Users/raymondbaranski/GitHub/PaperManager/resources'
		icon_names = [NO_STATUS, CONFIRMED, MATCH, NEAR_MATCH, BAD_BIBTEX, BAD_RECORDS, NO_RECORDS]
		self.bibtex_to_icon = dict()
		for icon_name in icon_names:
			self.bibtex_to_icon[icon_name] = PhotoImage(file=os.path.join(resource_folder, '{}.png'.format(icon_name)))

		self.pdf_text_width = 100

	def start(self):
		self.ui.root.mainloop()

	def delete_selected_document(self):
		selected_item = self.ui.document_listbox.selection()
		doc_id = self.ui.document_listbox.item(selected_item)['values'][0]
		if doc_id != 'None':
			document = self.id_to_doc[doc_id]
			old_file_path = document.filepath
			file_name = document.filename
			new_file_path = os.path.join(self.trash_directory, file_name)
			answer = askyesno(title='Delete', message='Are you sure you want to delete this document?\nIt will be moved to "{}"'.format(new_file_path))
			if answer:
				os.remove(os.path.join(self.main_directory, 'documents', '{}.docinf'.format(doc_id)))
				os.rename(old_file_path, new_file_path)
				self.documents.pop(old_file_path)
				self.id_to_doc.pop(doc_id)

				self.ui.document_listbox.delete(selected_item)

	def update_records(self, document, method):
		if method == MENDELEY:
			records = self.netscraper.mendeley_search(document.title)
		elif method == GOOGLE:
			records = self.netscraper.google_search(document.title)
		elif method == CROSSREF:
			records = self.netscraper.crossref_search(document.title)
		else:
			raise ValueError('<method> should be either {} or {}, not {}'.format(MENDELEY, GOOGLE, method))
		document.extend_records(records)
		bibtex_status = document.calculate_document_status()
		return bibtex_status

	# def rename_document(self):
	# 	if self.selected_document is not None:
	# 		old_filepath = self.selected_document.filepath
	# 		new_filename = self.selected_document.calc_safe_filename()
	# 		self.selected_document.rename_file(new_filename)
	# 		filepath = self.selected_document.filepath
	# 		self.documents[filepath] = self.documents.pop(old_filepath)

	def check_mendeley(self):
		if self.selected_document is not None:
			self.update_records(self.selected_document, MENDELEY)
			self.update_document_in_listbox()

	def bulk_check_mendeley(self):
		documents_to_check = list()
		for doc in self.documents.values():
			if doc.bibtex_status not in [CONFIRMED, MATCH, NEAR_MATCH]:
				documents_to_check.append(doc)

		# progress_var = DoubleVar()
		progress_bar_window = tk.Toplevel()
		progress_bar_window.wm_title('Checking Mendeley...')
		top_frame = Frame(progress_bar_window)
		top_frame.pack(side=TOP)

		progressbar = ttk.Progressbar(
			top_frame,
			value=0,
			orient='horizontal',
			mode='determinate',
			length=500,
			maximum=len(documents_to_check)
		)
		progressbar.pack(side=LEFT)
		doc_var = StringVar()
		prog_var = StringVar()

		progress_bar_label = Label(top_frame, height=2, width=10, textvariable=prog_var)
		progress_bar_label.pack(side=LEFT)
		doc_title_label = Label(progress_bar_window, height=4, width=100, textvariable=doc_var)
		doc_title_label.pack(side=TOP)

		progress_bar_window.update()
		progress_bar_window.update_idletasks()

		for i in range(len(documents_to_check)):
			doc = documents_to_check[i]
			progressbar.step(1)
			prog_var.set('{}/{}'.format(i+1, len(documents_to_check)))
			doc_var.set(doc.title)
			time.sleep(0.02)
			self.update_records(doc, MENDELEY)
			doc.save(self.documents_directory)
			# self.ui.root.update_idletasks()
			progress_bar_window.update()
			progress_bar_window.update_idletasks()
		progress_bar_window.destroy()
		self.display_docs(self.documents.values())

	def check_google(self):
		url = 'https://scholar.google.com/'
		if self.selected_document is not None:
			write_to_clipboard(self.selected_document.title)
			controller = webbrowser.get('Firefox')
			controller.open(url)
		# if self.selected_document is not None:
		# 	self.update_records(self.selected_document, GOOGLE)
		# 	self.update_document_in_listbox()
	def get_duplicate_documents_fast(self):
		"""
		Okay, here's the idea. Basically, we want to get all duplicates or potential duplicates 'together'. What does
		that mean? In theory, we could just use the bibtex. Probably, we should just use the bibtex.
		Sometimes, we don't have the bibtex
		Honestly, let's just use the title
		"""
		duplicates = dict()
		docs_dict = dict()
		for doc in self.documents.values():
			title = doc.title
			if title in docs_dict.keys():
				if title in duplicates.keys():
					duplicates[title].add(doc)
				else:
					duplicates[title] = set()
					duplicates[title].add(docs_dict[title])
					duplicates[title].add(doc)
			else:
				docs_dict[title] = doc
		return duplicates

	def check_crossref(self):
		if self.selected_document is not None:
			self.update_records(self.selected_document, CROSSREF)
			self.update_document_in_listbox()

	def update_document_in_listbox(self):
		selected_item = self.ui.document_listbox.selection()
		self.ui.document_listbox.set(selected_item, column=1, value=self.selected_document.title)

		icon = self.bibtex_to_icon[self.selected_document.bibtex_status]
		self.ui.document_listbox.item(selected_item, image=icon)

	def get_selected_documents(self):
		selected_items = self.ui.document_listbox.selection()
		doc_ids = [self.ui.document_listbox.item(i)['values'][0] for i in selected_items]
		documents = [self.id_to_doc[doc_id] for doc_id in doc_ids]
		return documents

	def select_document(self, event):
		if self.selected_document is not None:
			self.selected_document.update_notes(self.documents_directory, self.ui)

		selected_item = self.ui.document_listbox.selection()
		if len(selected_item) == 1:
			doc_id = self.ui.document_listbox.item(selected_item)['values'][0]
			if doc_id != 'None':
				# doc_id = document_listbox.item(selected_item)['text']
				self.selected_document = self.id_to_doc[doc_id]

				self.selected_document.load_pdf(self.ui)

				self.ui.info_panel.delete('1.0', END)
				self.ui.info_panel.insert(END, 'Title: {}'.format(self.selected_document.title))

				self.ui.tags_panel.config(state=NORMAL)
				self.ui.tags_panel.delete('1.0', END)
				self.ui.tags_panel.insert(END, self.selected_document.get_tags_string())
				self.ui.tags_panel.config(state=DISABLED)

				self.ui.bibtex.delete('1.0', END)
				if self.selected_document.bibtex is not None:
					self.ui.bibtex.insert(END, self.selected_document.bibtex)
				# info_panel.insert(END, 'Authors: {}'.format(self.authors))

				self.ui.notes.delete('1.0', END)
				if self.selected_document.notes is not None:
					self.ui.notes.insert(END, self.selected_document.notes)
				# document = self.documents[item['values'][1]]
				# print(document.filepath)

	def manual_update_document_info(self, event):
		if self.ui.info_panel.edit_modified():
			new_title_string = self.get_title_from_info_panel()
			if self.selected_document is not None:
				self.selected_document.title = new_title_string
				self.selected_document.saved = False
			self.ui.info_panel.edit_modified(0)
			self.update_document_in_listbox()

	def key_press(self, e):
		if e.keycode == 855638143:
			self.delete_selected_document()

	def save_bibtex(self):
		if self.selected_document is not None:
			new_bibtex = self.ui.bibtex.get('1.0', END)
			self.selected_document.bibtex = new_bibtex
			self.selected_document.bibtex_status = CONFIRMED
			self.selected_document.save(self.documents_directory)
			self.update_document_in_listbox()

	def use_bibtex_title(self):
		if self.selected_document is not None:
			try:
				bib_dict = bibtex_to_dict(self.selected_document.bibtex)
				title = title_from_bibtex(bib_dict)
				self.selected_document.title = title
				self.ui.info_panel.delete('1.0', END)
				self.ui.info_panel.insert(END, 'Title: {}'.format(title))
				self.ui.info_panel.edit_modified(0)
				self.update_document_in_listbox()
				self.selected_document.saved = False
			except Exception as e:
				print(e)

	def auto_accept(self):
		if self.selected_document is not None:
			if self.selected_document.bibtex_status == NEAR_MATCH:
				# find best record
				max_similarity = 0
				best_record = None
				for record in self.selected_document.records:
					if record['similarity'] > max_similarity:
						max_similarity = record['similarity']
						best_record = record
				if best_record is not None:
					self.selected_document.bibtex = best_record['bibtex']
					self.selected_document.bibtex_status = CONFIRMED
					self.ui.bibtex.delete('1.0', END)
					self.ui.bibtex.insert(END, self.selected_document.bibtex)
					self.use_bibtex_title()
					self.update_document_in_listbox()
					self.selected_document.saved = False
					self.selected_document.save(self.documents_directory)

	def get_title_from_info_panel(self):
		info_panel_string = self.ui.info_panel.get('1.0', END)
		info_panel_elements = info_panel_string.split('\n')
		title_element = None
		for info_panel_element in info_panel_elements:
			if 'Title: ' in info_panel_element:
				title_element = info_panel_element
				break
		title_string = title_element[7:]
		return title_string

	def open_records_window(self):
		if self.selected_document is not None:
			window = tk.Toplevel()
			window.wm_title('Records')
			cur_title_display = Text(window, relief=GROOVE, borderwidth=2, width=180, height=1, fg='red')
			cur_title_display.pack()
			rec_title_display = Text(window, relief=GROOVE, borderwidth=2, width=180, height=1)
			rec_title_display.pack()
			upper_frame = Frame(window)
			upper_frame.pack()
			self.selected_record_info = Text(upper_frame, relief=GROOVE, borderwidth=2, width=90, height=20)
			self.selected_record_info.pack(side=LEFT)
			self.selected_bibtex = Text(upper_frame, relief=GROOVE, borderwidth=2, width=90, height=20)
			self.selected_bibtex.pack(side=LEFT)
		else:
			return None

		def use_record():
			record_index = self.records_listbox.index(self.records_listbox.selection())
			record = self.selected_document.records[record_index]
			if 'bibtex' in record:
				self.selected_document.bibtex_status = CONFIRMED
				self.selected_document.bibtex = record['bibtex']
				self.ui.bibtex.delete('1.0', END)
				self.ui.bibtex.insert(END, self.selected_document.bibtex)
				self.selected_document.save(self.documents_directory)
				self.update_document_in_listbox()

		def get_bibtex():
			record_index = self.records_listbox.index(self.records_listbox.selection())
			record = self.selected_document.records[record_index]
			if 'doi' in record:
				bibtex = self.netscraper.get_bibtex(record['doi'])
				if bibtex is not None:
					record['bibtex'] = bibtex
					self.selected_bibtex.delete('1.0', END)
					self.selected_bibtex.insert(END, record['bibtex'])
					self.selected_document.save(self.documents_directory)

		button_frame = Frame(window)
		button_frame.pack(side=TOP)
		self.use_record_button = Button(button_frame, text='Use Record', command=use_record)
		self.use_record_button.pack(side=LEFT)
		self.get_bibtex_button = Button(button_frame, text='Get Bibtex', command=get_bibtex)
		self.get_bibtex_button.pack(side=LEFT)
		self.use_title_button = Button(button_frame, text='Use Title', command=self.use_bibtex_title)
		self.use_title_button.pack(side=LEFT)

		self.records_listbox = ttk.Treeview(window, column=('c1', 'c2', 'c3'), show='headings', height=10)
		self.records_listbox.tag_configure('light', background='#aaaaaa')
		self.records_listbox.tag_configure('gray', background='#cccccc')
		self.records_listbox.column("# 1", anchor=CENTER, stretch=NO, minwidth=100, width=100)
		self.records_listbox.heading("# 1", text="Match")
		self.records_listbox.column("# 2", anchor=W, stretch=NO, minwidth=600, width=600)
		self.records_listbox.heading("# 2", text="Title")
		self.records_listbox.column("# 3", anchor=W, stretch=NO, minwidth=600, width=600)
		self.records_listbox.heading("# 3", text="Authors")
		self.records_listbox.pack()
		sorted_records = sorted(self.selected_document.records, key=lambda x: -x['similarity'])
		self.selected_document.records = sorted_records
		for i in range(len(sorted_records)):
			record = sorted_records[i]
			if 'authors' in record:
				values = ('{:.2f}%'.format(record['similarity']*100), record['title'], record['authors'])
			elif 'author' in record:
				values = ('{:.2f}%'.format(record['similarity']*100), record['title'], record['author'])
			else:
				values = ('{:.2f}%'.format(record['similarity']*100), record['title'], 'No Authors')
			if i % 2 == 0:
				elem_tag = 'light'
			else:
				elem_tag = 'gray'
			self.records_listbox.insert('', END, text=i, values=values, tag=elem_tag)
		def refresh(event):
			self.selected_record_info.delete('1.0', END)
			record_index = self.records_listbox.index(self.records_listbox.selection())
			record = self.selected_document.records[record_index]
			record_string = ''
			# fields = ['title', 'authors', 'source', 'year', 'identifiers', 'keywords', 'link']
			fields = ['title', 'authors', 'author', 'source', 'journal', 'publisher', 'year', 'month', 'identifiers', 'doi', 'url', 'keywords', 'link']
			# fields = ['title', 'authors', 'author', 'source', 'journal', 'publisher', 'year', 'month', 'identifiers', 'doi', 'url', 'link']

			fields = sort_keys(list(record.keys()), fields, ['bibtex', 'similarity'])
			for field in fields:
				record_string += '{} : {}\n'.format(field, record[field])
			if len(record_string) > 1500:
				record_string = record_string[:1500]
			self.selected_record_info.insert(END, record_string)

			self.selected_bibtex.delete('1.0', END)
			if 'bibtex' in record:
				self.selected_bibtex.insert(END, record['bibtex'])

			cur_title_display.delete('1.0', END)
			cur_title_display.insert(END, clean_string(self.selected_document.title))
			rec_title_display.delete('1.0', END)
			rec_title_display.insert(END, clean_string(record['title']))

		self.records_listbox.bind('<<TreeviewSelect>>', refresh)

	def open_tags_interface(self):
		self.tag_manager.open_tag_window()

	def open_text_interface(self):
		text_window = tk.Toplevel()
		text_window.wm_title('Document Text')

		text_top_frame = Frame(text_window)
		text_top_frame.pack(side=TOP)

		get_text_button = Button(text_top_frame, text='Get Text', command=self.get_document_text)
		get_text_button.pack(side=LEFT)

		self.text_frame = Frame(text_window)
		self.text_frame.pack(side=TOP)
		self.pdf_text = Text(self.text_frame, height=60, width=self.pdf_text_width)
		self.pdf_text.pack(side=LEFT)
		self.text_scrollbar = Scrollbar(self.text_frame)
		self.text_scrollbar.pack(side=LEFT, fill=BOTH)
		self.pdf_text.config(yscrollcommand=self.text_scrollbar.set)
		self.text_scrollbar.config(command=self.pdf_text.yview)

		# self.doc_scrollbar = Scrollbar(self.canvas)
		# self.doc_scrollbar.pack(side=LEFT, fill=BOTH)
		# self.document_listbox.config(yscrollcommand=self.doc_scrollbar.set)
		# self.doc_scrollbar.config(command=self.document_listbox.yview)

		if self.selected_document.raw_text is not None:
			self.pdf_text.config(state=NORMAL)
			self.pdf_text.insert(END, organize_raw_text(self.selected_document.raw_text, self.pdf_text_width))
			self.pdf_text.config(state=DISABLED)

	def get_document_text(self):
		if self.selected_document is not None:
			self.selected_document.load_text(TEXTRACT)
			self.pdf_text.config(state=NORMAL)
			self.pdf_text.delete('1.0', END)
			self.pdf_text.insert(END, organize_raw_text(self.selected_document.raw_text, self.pdf_text_width))
			self.pdf_text.config(state=DISABLED)
			self.selected_document.save(self.documents_directory)

	def open_export_interface(self):
		self.export_manager.open_export_interface()

	def insert_document(self, i, doc, index):
		values = (doc.document_id, doc.title, '')
		if i % 2 == 0:
			tag = 'light'
		else:
			tag = 'gray'
		image = self.bibtex_to_icon[doc.bibtex_status]
		self.ui.document_listbox.insert('', index, image=image, values=values, tag=tag)

	def display_docs(self, doc_list):
		self.ui.document_listbox.delete(*self.ui.document_listbox.get_children())
		i = 0
		for doc in doc_list:
			self.insert_document(i, doc, END)
			i += 1
		self.ui.document_listbox.tag_configure('light', background='#aaaaaa')
		self.ui.document_listbox.tag_configure('gray', background='#cccccc')

	def insert_duplicate_title(self, iid, title, index):
		values = (None, title, '')
		self.ui.document_listbox.insert('', index, iid=iid, values=values)

	def insert_duplicate_document(self, i, iid, doc, index):
		values = (doc.document_id, doc.title, '')
		if i % 2 == 0:
			tag = 'light'
		else:
			tag = 'gray'
		image = self.bibtex_to_icon[doc.bibtex_status]
		self.ui.document_listbox.insert(iid, index, image=image, values=values, tag=tag)

	def display_duplicates(self):
		duplicates = self.get_duplicate_documents_fast()
		self.ui.document_listbox.delete(*self.ui.document_listbox.get_children())
		i = 0
		iid = 0
		for title in duplicates.keys():
			self.insert_duplicate_title(iid, title, END)
			for doc in duplicates[title]:
				self.insert_duplicate_document(i, iid, doc, END)
				i += 1
			iid += 1
		self.ui.document_listbox.tag_configure('light', background='#aaaaaa')
		self.ui.document_listbox.tag_configure('gray', background='#cccccc')


	def search(self):
		self.selected_document = None
		search_text = self.ui.search_box.get('1.0', END)
		search_text = re.sub(r"[\n\t\s]*", "", search_text)
		if 'status:' in search_text:
			status_opts = search_text.replace('status:', '').split(',')
			results = list()
			for doc in self.documents.values():
				if doc.bibtex_status in status_opts:
					results.append(doc)
			self.display_docs(results)
		elif 'words:' in search_text:
			contains_opts = search_text.replace('words:', '')
			check_func = parse_word_search(contains_opts, 'doc.words')
			results = list()
			for doc in self.documents.values():
				if check_func(doc):
					results.append(doc)
			self.display_docs(results)
		elif 'text:' in search_text:
			contains_opts = search_text.replace('text:', '')
			check_func = parse_word_search(contains_opts, 'doc.raw_text')
			results = list()
			for doc in self.documents.values():
				if check_func(doc):
					results.append(doc)
			self.display_docs(results)
		elif 'tags:' in search_text:
			contains_opts = search_text.replace('tags:', '')
			check_func = parse_word_search(contains_opts, '[tag.tagword.lower() for tag in doc.tags]')
			results = list()
			for doc in self.documents.values():
				if check_func(doc):
					results.append(doc)
			self.display_docs(results)
		else:
			if search_text == '':
				self.display_docs(self.documents.values())
			else:
				results = list()
				for doc in self.documents.values():
					# print(doc.title.lower())
					if search_text.lower() in doc.title.lower():
						results.append(doc)
				self.display_docs(results)

	def display_search_help(self):
		help_window = tk.Toplevel()
		help_window.wm_title('Search Help')
		help_text = 'status:{}'.format(STATUS_OPTS)
		help_label = Text(help_window, height=10, width=40, wrap=WORD)
		help_label.insert(END, help_text)
		help_label.config(state=DISABLED)
		help_label.pack()

	def reload_files(self):
		files = os.listdir(self.documents_directory)
		files_loaded = False
		for file in files:
			if '.docinf' in file:
				files_loaded = True
				with open(os.path.join(self.documents_directory, file), 'rb') as f:
					loaded_doc = pickle.load(f)
					self.documents[loaded_doc.filepath] = loaded_doc
					self.id_to_doc[loaded_doc.document_id] = loaded_doc
		if files_loaded:
			self.display_docs(self.documents.values())

	def load_data(self):
		if os.path.exists(os.path.join(self.main_directory, 'data.json')):
			with open(os.path.join(self.main_directory, 'data.json'), 'r') as f:
				json_data = json.load(f)
			self.doc_id_counter = json_data['doc_id_counter']
			self.folder_watcher.watched_folders = set(json_data['watched_folders'])

		if os.path.exists(os.path.join(self.main_directory, 'dictionary.dat')):
			with open(os.path.join(self.main_directory, 'dictionary.dat'), 'rb') as f:
				dictionary_data = pickle.load(f)
				self.dictionary.load_data(dictionary_data)

		self.reload_files()
		self.load_tag_data()
		self.reconstruct_tag_data()

		# print(self.documents.keys())

	def load_tag_data(self):
		if os.path.exists(os.path.join(self.main_directory, 'tags.dat')):
			with open(os.path.join(self.main_directory, 'tags.dat'), 'rb') as f:
				tag_data = pickle.load(f)
				self.tag_manager.load_tag_data(tag_data)

	def reconstruct_tag_data(self):
		for doc in self.documents.values():
			tag_strings = doc.tags
			doc.tags = set([self.tag_manager.tags[tag_string] for tag_string in tag_strings])

	def save_metadata(self):
		json_data = {
			'doc_id_counter': self.doc_id_counter,
			'watched_folders': list(self.folder_watcher.watched_folders),
		}
		with open(os.path.join(self.main_directory, 'data.json'), 'w') as f:
			json.dump(json_data, f)

		dictionary_data = self.dictionary.get_data()

		with open(os.path.join(self.main_directory, 'dictionary.dat'), 'wb') as f:
			pickle.dump(dictionary_data, f)

	def save_tag_data(self):
		with open(os.path.join(self.main_directory, 'tags.dat'), 'wb') as f:
			pickle.dump(self.tag_manager.tags, f)

	def save_data(self):
		self.save_metadata()

		self.save_tag_data()

		for doc in self.documents.values():
			if not doc.saved:
				doc.save(self.documents_directory)

	def rename_file(self):
		if self.selected_document is not None:
			new_filename = self.selected_document.calc_safe_filename()
			if self.selected_document.check_if_filename_available(new_filename):
				old_filepath = self.selected_document.filepath
				new_filepath = os.path.join(self.selected_document.folder, new_filename)
				self.documents[new_filepath] = self.documents.pop(old_filepath)
				self.selected_document.rename_file(new_filename)
				self.selected_document.save(self.documents_directory)
			else:
				print('"{}" is already taken'.format(new_filename))

	def find_document_keywords(self):
		if self.selected_document is not None:
			self.selected_document.calculate_keywords(self.dictionary)
			self.ui.keywords_panel.delete('1.0', END)
			self.ui.keywords_panel.insert(END, ', '.join(self.selected_document.keywords))

	def find_keywords(self):
		for doc in self.documents.values():
			doc.calculate_keywords(self.dictionary)
		if self.selected_document is not None:
			self.ui.keywords_panel.delete('1.0', END)
			self.ui.keywords_panel.insert(END, ', '.join(self.selected_document.keywords))

	def open_document(self):
		if self.selected_document is not None:
			opener = "open" if sys.platform == "darwin" else "xdg-open"
			subprocess.call([opener, self.selected_document.filepath])
			# print('Opened {}'.format(self.selected_document.title))


class UIManager(object):
	def __init__(self, ref_manager):
		self.ref_manager = ref_manager

		self.root = tk.Tk()
		self.root.configure(bg=rgbtohex((.5, .5, .5)))
		self.canvas = tk.Canvas(self.root, bg='black', width=1600, height=900)
		self.canvas.pack()

		self.top_frame = Frame(self.canvas)
		self.top_frame.pack(side=TOP, fill=BOTH, expand=TRUE)

		self.search_box = Text(self.top_frame, width=100, height=1, borderwidth=2, relief=GROOVE)
		self.search_box.pack(in_=self.top_frame, side=LEFT)
		self.search_button = Button(text='Search', command=self.ref_manager.search)
		self.search_button.pack(in_=self.top_frame, fill=Y, ipady=5, ipadx=5, side=LEFT)
		self.search_help_button = Button(text='Search Help', command=self.ref_manager.display_search_help)
		self.search_help_button.pack(in_=self.top_frame, fill=Y, ipady=5, ipadx=5, side=LEFT)

		# search_box.insert(END, 'some text')

		self.left_frame = Frame(self.canvas)
		self.left_frame.pack(side=LEFT, fill=BOTH)

		# document_listbox = Listbox(left_frame, width=100, height=50)
		self.document_listbox = ttk.Treeview(self.left_frame, column=('c1', 'c2'), height=45)
		self.document_listbox.column('#0', anchor=CENTER, stretch=NO, minwidth=50, width=50)
		self.document_listbox.column("#1", anchor=CENTER, stretch=NO, minwidth=100, width=100)
		self.document_listbox.heading("#1", text="Id")
		self.document_listbox.column("#2", anchor=W, stretch=NO, minwidth=600, width=600)
		self.document_listbox.heading("#2", text="Title")
		self.document_listbox.pack(in_=self.left_frame, side=TOP, expand=TRUE)
		self.document_listbox.bind('<<TreeviewSelect>>', self.ref_manager.select_document)
		self.document_listbox.bind('<KeyPress>', self.ref_manager.key_press)

		self.doc_scrollbar = Scrollbar(self.canvas)
		self.doc_scrollbar.pack(side=LEFT, fill=BOTH)
		self.document_listbox.config(yscrollcommand=self.doc_scrollbar.set)
		self.doc_scrollbar.config(command=self.document_listbox.yview)

		self.right_frame = Frame(self.canvas)
		self.right_frame.pack(side=RIGHT, fill=BOTH)

		self.sidebar = Frame(self.right_frame, borderwidth=2, relief=GROOVE)
		self.sidebar.pack(side=LEFT, fill=BOTH, expand=TRUE)

		self.info_panel = Text(self.sidebar, relief=GROOVE, borderwidth=2, width=40, height=7, wrap=WORD)
		self.info_panel.pack(side=TOP, expand=TRUE)
		self.info_panel.bind('<<Modified>>', self.ref_manager.manual_update_document_info)

		self.open_frame = Frame(self.sidebar)
		self.open_frame.pack(side=TOP)
		self.open_pdf_button = Button(self.open_frame, text='Open PDF', command=self.ref_manager.open_document)
		self.open_pdf_button.pack(side=LEFT)
		self.open_text_button = Button(self.open_frame, text='Open Text', command=self.ref_manager.open_text_interface)
		self.open_text_button.pack(side=LEFT)

		self.tags_panel = Text(self.sidebar, relief=GROOVE, borderwidth=2, width=40, height=4, wrap=WORD)
		self.tags_panel.pack(side=TOP, expand=TRUE)

		self.bibtex = Text(self.sidebar, relief=GROOVE, borderwidth=2, width=40, height=20, wrap=WORD)
		self.bibtex.pack(side=TOP, expand=TRUE)

		self.under_bibtex_frame = Frame(self.sidebar)
		self.under_bibtex_frame.pack(side=TOP)
		self.save_bibtex_button = Button(self.under_bibtex_frame, text='Save Bibtex', command=self.ref_manager.save_bibtex)
		self.save_bibtex_button.pack(side=LEFT)
		self.use_title_button = Button(self.under_bibtex_frame, text='Use Title', command=self.ref_manager.use_bibtex_title)
		self.use_title_button.pack(side=LEFT)
		self.rename_file_button = Button(self.under_bibtex_frame, text='Rename', command=self.ref_manager.rename_file)
		self.rename_file_button.pack(side=LEFT)

		self.notes = Text(self.sidebar, relief=GROOVE, borderwidth=2, width=40, height=15, wrap=WORD)
		self.notes.pack(side=TOP, expand=TRUE)

		self.netscrape_frame = Frame(self.sidebar)
		self.netscrape_frame.pack(side=TOP)
		self.check_mendeley_button = Button(self.netscrape_frame, text='Mendeley', command=self.ref_manager.check_mendeley)
		self.check_mendeley_button.pack(side=LEFT)
		self.check_crossref_button = Button(self.netscrape_frame, text='Crossref', command=self.ref_manager.check_crossref)
		self.check_crossref_button.pack(side=LEFT)
		self.check_google_button = Button(self.netscrape_frame, text='Google', command=self.ref_manager.check_google)
		self.check_google_button.pack(side=LEFT)
		self.records_frame = Frame(self.sidebar)
		self.records_frame.pack(side=TOP)
		self.see_records_button = Button(self.records_frame, text='Records', command=self.ref_manager.open_records_window)
		self.see_records_button.pack(side=LEFT)
		self.accept_record_button = Button(self.records_frame, text='Accept', command=self.ref_manager.auto_accept)
		self.accept_record_button.pack(side=LEFT)

		self.pdf_frame = Frame(self.right_frame)
		self.pdf_frame.pack(side=LEFT, fill=BOTH)
		self.scrol_y = Scrollbar(self.pdf_frame, orient=VERTICAL)
		self.pdf = Text(self.pdf_frame, yscrollcommand=self.scrol_y.set, bg="grey")
		self.scrol_y.pack(side=RIGHT, fill=Y)
		self.scrol_y.config(command=self.pdf.yview)
		self.pdf.pack(fill=BOTH, expand=1)

		# pdf_ids = list()
		# filepath = '/Users/raymondbaranski/Downloads/Distributed_intelligent_planning_and_scheduling_DI.pdf'
		# pages = convert_from_path(filepath, size=(800, 900))
		#
		# photos = []
		# for i in range(len(pages)):
		# 	photos.append(ImageTk.PhotoImage(pages[i]))
		# for photo in photos:
		# 	pdf_ids.append(pdf.image_create(END, image=photo))
		# 	pdf.insert(END, '\n\n')

		self.bottom_frame = Frame(self.root)
		self.bottom_frame.pack(side=BOTTOM, fill=BOTH, expand=TRUE)

		self.folder_manager_button = Button(text='Folders', command=self.ref_manager.folder_watcher.open_folder_window)
		self.folder_manager_button.pack(in_=self.bottom_frame, fill=Y, ipady=5, ipadx=5, side=LEFT)

		self.save_button = Button(text='Save', command=self.ref_manager.save_data)
		self.save_button.pack(in_=self.bottom_frame, fill=Y, ipady=5, ipadx=5, side=LEFT)

		self.tags_button = Button(self.bottom_frame, text='Tags', command=self.ref_manager.open_tags_interface)
		self.tags_button.pack(in_=self.bottom_frame, fill=Y, ipady=5, ipadx=5, side=LEFT)

		self.dupes_button = Button(self.bottom_frame, text='Duplicates', command=self.ref_manager.display_duplicates)
		self.dupes_button.pack(in_=self.bottom_frame, fill=Y, ipady=5, ipadx=5, side=LEFT)

		self.get_text_button = Button(text='Load Text', command=self.ref_manager.dictionary.parallel_load_text)
		self.get_text_button.pack(in_=self.bottom_frame, fill=Y, ipady=5, ipadx=5, side=LEFT)

		self.export_button = Button(self.bottom_frame, text='Export', command=self.ref_manager.open_export_interface)
		self.export_button.pack(in_=self.bottom_frame, fill=Y, ipady=5, ipadx=5, side=LEFT)

		self.bulk_scrape_button = Button(text='Bulk Scrape', command=self.ref_manager.netscraper.bulk_scrape_interface)
		self.bulk_scrape_button.pack(in_=self.bottom_frame, fill=Y, ipady=5, ipadx=5, side=RIGHT)

# ref_manager = RefManager()



# document_listbox.delete(0, END)



# ref_manager.load_data()

# root.mainloop()

if __name__ == '__main__':
	ref_manager = RefManager()
	ref_manager.load_data()
	ref_manager.folder_watcher.check_for_new_documents()
	# ref_manager.dictionary.parallel_calculate_dictionary()
	# ref_manager.dictionary.get_full_word_information()
	ref_manager.start()