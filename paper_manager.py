import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
from mendeley import Mendeley
import requests
import sys
import urllib.request
from urllib.error import HTTPError
import ssl
import os
import re
from pdf2image import convert_from_path
import pdfplumber
import matplotlib.pyplot as plt
import numpy as np
import time
from collections import defaultdict
import json
import pickle
import Levenshtein as lev
from tkinter import PhotoImage
from gscholar import query


NO_STATUS = 'no_status'  # we haven't requested records yet [gray circle]
CONFIRMED = 'confirmed'  # user has confirmed bibtex [blue circle]
MATCH = 'match'  # there is a match with good bibtex [green circle]
NEAR_MATCH = 'near_match'  # there are near-matches with good bibtex [green exclamation point]
BAD_BIBTEX = 'bad_bibtex'  # there are matches or near-matches with bad or NO bibtex [yellow exclamation point]
BAD_RECORDS = 'bad_records'  # there are records, but there isn't a good match [red circle]
NO_RECORDS = 'no_records'  # there are simply no records at all [red exclamation point]

MENDELEY = 'mendeley'
GOOGLE = 'google'

# there are
# there are records, but they aren't matches
# there are no records


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


def get_similarity_score(res_title, doc_title):
	"""
	Calculates the Levenshtein ratio between two titles
	:param res_title:
	:param doc_title:
	:return:
	"""
	res_title = res_title.lower()
	cur_title = doc_title.lower()
	res_title = ''.join(e for e in res_title if e.isalnum())
	cur_title = ''.join(e for e in cur_title if e.isalnum())
	similarity = lev.ratio(res_title, cur_title)
	return similarity


def bibtex_quality_score(bibtex):
	return 1


def bibtex_to_dict(bibtex_string):
	bibtex_fields = [bibtex_field for bibtex_field in bibtex_string.split('\n') if '=' in bibtex_field]
	bibdict = dict()
	for field in bibtex_fields:
		key, value = field.split('=')
		key = key.replace(' ', '')
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

		self.infolist = list()

		self.bibtex = None
		self.raw_text = None
		self.raw_words = dict()
		self.words = list()
		self.word_counts = dict()
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
		return state


	def load_first_pages(self):
		factor = 550 / 800
		if len(self.photos) < 2:
			pages = convert_from_path(self.filepath, size=(800 * factor, 1000 * factor), last_page=2)
			for i in range(len(pages)):
				self.photos.append(ImageTk.PhotoImage(pages[i]))

	def load_pdf(self):
		pdf.delete('1.0', END)

		self.load_first_pages()

		for photo in self.photos:
			self.pdf_ids.append(pdf.image_create(END, image=photo))
			pdf.insert(END, '\n\n')

	def load_text(self):
		self.saved = False
		text = ''
		with pdfplumber.open(self.filepath) as pdf_file:
			for page in pdf_file.pages:
				page_text = page.extract_text(x_tolerance=self.x_tolerance)
				if page_text is not None:
					text += page_text
		dictionary = defaultdict(lambda: 0)
		text = re.sub('[\n()\[\]{}0-9&@…#%]', '', text)
		text = re.sub('[.,;:?…“”\'‘’/\"]', ' ', text)
		self.raw_text = text
		# text = text.replace('\n', '')
		# text = text.replace('.', ' ')

		words = text.split(' ')
		words = [word.lower() for word in words if len(word) >= 3]
		for word in words:
			dictionary[word] += 1
		self.raw_words = dict(dictionary)
		words = list(dictionary.keys())
		counts = list(dictionary.values())
		sort_idxs = np.flip(np.argsort(counts))
		# print(sort_idxs)
		words = [words[i] for i in sort_idxs]
		counts = [counts[i] for i in sort_idxs]

		dict_string = ['{} : {}'.format(words[i], counts[i]) for i in range(len(words))]
		# words = sorted(list(dictionary.keys()), key=len)
		# words = ['>' + word + '<' for word in words]
		# word_list =

		# pdf.delete('1.0', END)
		# pdf.insert('1.0', '\n'.join(dict_string))

		# pdf.insert('1.0', text)

	def update_notes(self, directory):
		new_text = notes.get('1.0', END).rstrip('\n')
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
	def __init__(self):
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


class RefManager(object):
	def __init__(self):
		self.doc_id_counter = 0
		self.watched_folders = set()
		self.documents = dict()
		self.id_to_doc = dict()
		self.folder_listbox = None
		self.selected_document = None

		self.all_words = defaultdict(lambda: 0)
		self.keywords = list()
		self.normwords = list()
		self.trashwords = list()
		self.unsorted = list()

		self.main_directory = '/Users/raymondbaranski/.paper_manager'
		if not os.path.exists(self.main_directory):
			os.mkdir(self.main_directory)
		self.documents_directory = os.path.join(self.main_directory, 'documents')
		if not os.path.exists(self.documents_directory):
			os.mkdir(self.documents_directory)

		resource_folder = '/Users/raymondbaranski/GitHub/mendeley-api-python-example/resources'
		icon_names = [NO_STATUS, CONFIRMED, MATCH, NEAR_MATCH, BAD_BIBTEX, BAD_RECORDS, NO_RECORDS]
		self.bibtex_to_icon = dict()
		for icon_name in icon_names:
			self.bibtex_to_icon[icon_name] = PhotoImage(file=os.path.join(resource_folder, '{}.png'.format(icon_name)))

		self.netscraper = NetScraper()

	def update_records(self, document, method):
		if method == MENDELEY:
			records = self.netscraper.mendeley_search(document.title)
		elif method == GOOGLE:
			records = self.netscraper.google_search(document.title)
		else:
			raise ValueError('<method> should be either {} or {}, not {}'.format(MENDELEY, GOOGLE, method))
		document.extend_records(records)
		bibtex_status = document.calculate_document_status()
		return bibtex_status

	def mendeley_search(self, title):
		results = self.session.catalog.search(title)
		max_iter = 10
		index = 0
		self.selected_document.records.clear()
		for doc in results.iter():
			index += 1
			result = self._get_mendeley_record(doc)
			if result['similarity'] == 1:
				if 'bibtex' in result:
					self.selected_document.bibtex_status = 'unconfirmed'
			self.selected_document.records.append(result)
			if index >= max_iter:
				break
		self.calculate_document_status()

	def calculate_document_status(self):
		# if there is a confirmed record or a 100% match, we do nothing
		if self.selected_document.bibtex_status not in ['unconfirmed', 'confirmed']:
			# if there are any records at all
			if len(self.selected_document.records) > 0:
				# did any records have bibtex and high similarity? Maybe just a mispelled title
				if any(['bibtex' in rec and rec['similarity'] > .9 for rec in self.selected_document.records]):
					self.selected_document.bibtex_status = 'mispell'
				else:
					self.selected_document.bibtex_status = 'no matches'
			else:
				self.selected_document.bibtex_status = 'no records'
		self.update_document_in_listbox()

	def get_similarity_score(self, res_title):
		res_title = res_title.lower()
		cur_title = self.selected_document.title.lower()
		res_title = ''.join(e for e in res_title if e.isalnum())
		cur_title = ''.join(e for e in cur_title if e.isalnum())
		similarity = lev.ratio(res_title, cur_title)
		return similarity

	def google_search(self, title):
		results = query(title)
		self.selected_document.records.clear()
		for doc in results:
			result = self._get_google_record(doc)
			if result['similarity'] == 1:
				if 'bibtex' in result:
					self.selected_document.bibtex_status = 'unconfirmed'
			self.selected_document.records.append(result)
		self.calculate_document_status()

	def _get_google_record(self, result):
		results_dict = bibtex_to_dict(result)
		similarity = self.get_similarity_score(results_dict['title'])
		results_dict['similarity'] = similarity
		results_dict['bibtex'] = result
		return results_dict

	def _get_mendeley_record(self, result):
		field_names = ['title', 'authors', 'identifiers', 'keywords', 'link', 'source', 'year']
		max_field_len = max([len(field_name) for field_name in field_names])
		results_dict = dict()
		results_string = ''

		# set to lower case and remove special characters
		similarity = self.get_similarity_score(result.title)

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

	def get_authors_pretty(self, author_list):
		if isinstance(author_list, list):
			return ['{} {}'.format(author.first_name, author.last_name) for author in author_list]
		else:
			if author_list is not None:
				return '{} {}'.format(author_list.first_name, author_list.last_name)
			else:
				return None

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

		self.save_metadata()

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
		calc_dictionary_button = Button(window, text='Calculate', command=self.calculate_dictionary)
		calc_dictionary_button.pack()

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

	def update_word_list(self, listbox, word_list):
		counts = [self.all_words[word] for word in word_list]
		sort_idxs = np.flip(np.argsort(counts))
		words = [word_list[i] for i in sort_idxs]
		counts = [counts[i] for i in sort_idxs]

		listbox.delete(*listbox.get_children())
		for i in range(len(words)):
			# print(words[i], counts[i])
			listbox.insert('', END, text=words[i], values=(words[i], counts[i]))

	def calculate_dictionary(self):
		self.all_words = defaultdict(lambda: 0)
		for doc in self.documents.values():
			print(doc.title)
			if len(doc.raw_words) == 0:
				doc.load_text()
			# print(doc.raw_words)
			for word in doc.raw_words.keys():
				self.all_words[word] += doc.raw_words[word]
		# print('\n\n')

		for word in self.all_words.keys():
			if not (word in self.keywords or word in self.normwords or word in self.trashwords or word in self.unsorted):
				if self.all_words[word] == 1:
					self.trashwords.append(word)
				else:
					self.unsorted.append(word)
		self.update_word_lists()

		# words = list(self.all_words.keys())
		# counts = list(self.all_words.values())
		# sort_idxs = np.flip(np.argsort(counts))
		# words = [words[i] for i in sort_idxs]
		# counts = [counts[i] for i in sort_idxs]

		# for i in range(len(words)):
		# 	print('{} : {}'.format(words[i], counts[i]))

	def load_data(self):
		if os.path.exists(os.path.join(self.main_directory, 'data.json')):
			with open(os.path.join(self.main_directory, 'data.json'), 'r') as f:
				json_data = json.load(f)
			self.doc_id_counter = json_data['doc_id_counter']
			self.watched_folders = list(json_data['watched_folders'])

		if os.path.exists(os.path.join(self.main_directory, 'dictionary.dat')):
			with open(os.path.join(self.main_directory, 'dictionary.dat'), 'rb') as f:
				dictionary_data = pickle.load(f)
				self.all_words = defaultdict(int, dictionary_data['all_words'])
				self.keywords = dictionary_data['keywords']
				self.normwords = dictionary_data['normwords']
				self.trashwords = dictionary_data['trashwords']
				self.unsorted = dictionary_data['unsorted']

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
			self.reset_doclist()

	def save_metadata(self):
		json_data = {
			'doc_id_counter': self.doc_id_counter,
			'watched_folders': list(self.watched_folders),
		}
		with open(os.path.join(self.main_directory, 'data.json'), 'w') as f:
			json.dump(json_data, f)

		dictionary_data = {
			'all_words': dict(self.all_words),
			'keywords': self.keywords,
			'normwords': self.normwords,
			'trashwords': self.trashwords,
			'unsorted': self.unsorted
		}

		with open(os.path.join(self.main_directory, 'dictionary.dat'), 'wb') as f:
			pickle.dump(dictionary_data, f)

	def save_data(self):
		self.save_metadata()

		for doc in self.documents.values():
			if not doc.saved:
				doc.save(self.documents_directory)

	def get_document_text(self):
		if self.selected_document is not None:
			self.selected_document.load_text()

	def update_document_in_listbox(self):
		selected_item = document_listbox.selection()
		document_listbox.set(selected_item, column=1, value=self.selected_document.title)

		icon = self.bibtex_to_icon[self.selected_document.bibtex_status]
		document_listbox.item(selected_item, image=icon)

	def select_document(self, event):
		if self.selected_document is not None:
			self.selected_document.update_notes(self.documents_directory)

		selected_item = document_listbox.selection()
		doc_id = document_listbox.item(selected_item)['values'][0]
		# doc_id = document_listbox.item(selected_item)['text']
		self.selected_document = self.id_to_doc[doc_id]

		self.selected_document.load_pdf()

		info_panel.delete('1.0', END)
		info_panel.insert(END, 'Title: {}'.format(self.selected_document.title))
		# info_panel.insert(END, 'Authors: {}'.format(self.authors))

		notes.delete('1.0', END)
		if self.selected_document.notes is not None:
			notes.insert(END, self.selected_document.notes)
		# document = self.documents[item['values'][1]]
		# print(document.filepath)

	def manual_update_document_info(self, event):
		if info_panel.edit_modified():
			new_title_string = self.get_title_from_info_panel()
			if self.selected_document is not None:
				self.selected_document.title = new_title_string
			info_panel.edit_modified(0)
			self.update_document_in_listbox()

	def get_title_from_info_panel(self):
		info_panel_string = info_panel.get('1.0', END)
		info_panel_elements = info_panel_string.split('\n')
		title_element = None
		for info_panel_element in info_panel_elements:
			if 'Title: ' in info_panel_element:
				title_element = info_panel_element
				break
		title_string = title_element[7:]
		return title_string

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

	def open_records_window(self):
		window = tk.Toplevel()
		window.wm_title('Records')
		upper_frame = Frame(window)
		upper_frame.pack()
		self.selected_record_info = Text(upper_frame, relief=GROOVE, borderwidth=2, width=90, height=20)
		self.selected_record_info.pack(side=LEFT)
		self.selected_bibtex = Text(upper_frame, relief=GROOVE, borderwidth=2, width=90, height=20)
		self.selected_bibtex.pack(side=LEFT)
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
		for i in range(len(self.selected_document.records)):
			record = self.selected_document.records[i]
			if 'authors' in record:
				values = ('{:.2f}%'.format(record['similarity']*100), record['title'], record['authors'])
			else:
				values = ('{:.2f}%'.format(record['similarity'] * 100), record['title'], record['author'])
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
			fields = sort_keys(list(record.keys()), fields, ['bibtex', 'similarity'])
			for field in fields:
				record_string += '{} : {}\n'.format(field, record[field])
			self.selected_record_info.insert(END, record_string)

			self.selected_bibtex.delete('1.0', END)
			if 'bibtex' in record:
				self.selected_bibtex.insert(END, record['bibtex'])

		self.records_listbox.bind('<<TreeviewSelect>>', refresh)

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
				for key, doc in self.documents.items():
					if doc.folder == folder:
						rm_docs.append(key)
				for key in rm_docs:
					del self.documents[key]
			self.update_folder_list()
			self.reset_doclist()
		except:
			pass

	def add_folder(self):
		foldername = filedialog.askdirectory(initialdir='/Users')
		if foldername != '':
			if foldername not in self.watched_folders:
				self.watched_folders.add(foldername)
				pdfs = get_pdfs_in_folder(foldername)
				for pdf in pdfs:
					if pdf not in self.documents.keys():
						new_doc = Document(pdf, self.doc_id_counter)
						self.doc_id_counter += 1
						self.documents[pdf] = new_doc
						self.id_to_doc[new_doc.document_id] = new_doc
		self.reset_doclist()
		self.update_folder_list()

	def insert_document(self, i, doc, index):
		values = (doc.document_id, doc.title, '')
		if i % 2 == 0:
			tag = 'light'
		else:
			tag = 'gray'
		image = self.bibtex_to_icon[doc.bibtex_status]
		document_listbox.insert('', index, image=image, values=values, tag=tag)

	def reset_doclist(self):
		document_listbox.delete(*document_listbox.get_children())
		i = 0
		for doc in self.documents.values():
			self.insert_document(i, doc, END)
			i += 1
		document_listbox.tag_configure('light', background='#aaaaaa')
		document_listbox.tag_configure('gray', background='#cccccc')

	def display_docs(self, doc_list):
		document_listbox.delete(*document_listbox.get_children())
		i = 0
		for doc in doc_list:
			self.insert_document(i, doc, END)
			i += 1
		document_listbox.tag_configure('light', background='#aaaaaa')
		document_listbox.tag_configure('gray', background='#cccccc')

	def search(self):
		search_text = search_box.get('1.0', END)
		search_text = re.sub(r"[\n\t\s]*", "", search_text)
		if search_text == '':
			self.reset_doclist()
		else:
			results = list()
			for doc in self.documents.values():
				# print(doc.title.lower())
				if search_text.lower() in doc.title.lower():
					results.append(doc)
			self.display_docs(results)

	def check_mendeley(self):
		if self.selected_document is not None:
			self.update_records(self.selected_document, MENDELEY)

	def check_google(self):
		if self.selected_document is not None:
			self.update_records(self.selected_document, GOOGLE)


root = tk.Tk()
root.configure(bg=rgbtohex((.5, .5, .5)))
canvas = tk.Canvas(root, bg='black', width=1600, height=900)
canvas.pack()

ref_manager = RefManager()

top_frame = Frame(canvas)
top_frame.pack(side=TOP, fill=BOTH, expand=TRUE)

search_box = Text(top_frame, width=50, height=1, borderwidth=2, relief=GROOVE)
search_box.pack(in_=top_frame, side=LEFT)
search_button = Button(text='Search', command=ref_manager.search)
search_button.pack(in_=top_frame, fill=Y, ipady=5, ipadx=5, side=LEFT)
# search_box.insert(END, 'some text')

left_frame = Frame(canvas)
left_frame.pack(side=LEFT, fill=BOTH)

# document_listbox = Listbox(left_frame, width=100, height=50)
document_listbox = ttk.Treeview(left_frame, column=('c1', 'c2'), height=45)
document_listbox.column('#0', anchor=CENTER, stretch=NO, minwidth=50, width=50)
document_listbox.column("#1", anchor=CENTER, stretch=NO, minwidth=100, width=100)
document_listbox.heading("#1", text="Id")
document_listbox.column("#2", anchor=W, stretch=NO, minwidth=600, width=600)
document_listbox.heading("#2", text="Title")
document_listbox.pack(in_=left_frame, side=TOP, expand=TRUE)
document_listbox.bind('<<TreeviewSelect>>', ref_manager.select_document)

# document_listbox.delete(0, END)

doc_scrollbar = Scrollbar(canvas)
doc_scrollbar.pack(side=LEFT, fill=BOTH)
document_listbox.config(yscrollcommand=doc_scrollbar.set)
doc_scrollbar.config(command=document_listbox.yview)

right_frame = Frame(canvas)
right_frame.pack(side=RIGHT, fill=BOTH)

sidebar = Frame(right_frame, borderwidth=2, relief=GROOVE)
sidebar.pack(side=LEFT, fill=BOTH, expand=TRUE)

info_panel = Text(sidebar, relief=GROOVE, borderwidth=2, width=40, height=10, wrap=WORD)
info_panel.pack(side=TOP, expand=TRUE)
info_panel.bind('<<Modified>>', ref_manager.manual_update_document_info)

notes = Text(sidebar, relief=GROOVE, borderwidth=2, width=40, height=20)
notes.pack(side=TOP, expand=TRUE)

check_mendeley_button = Button(sidebar, text='Check Mendeley', command=ref_manager.check_mendeley)
check_mendeley_button.pack(side=TOP)
check_google_button = Button(sidebar, text='Check Google', command=ref_manager.check_google)
check_google_button.pack(side=TOP)
see_records_button = Button(sidebar, text='See Records', command=ref_manager.open_records_window)
see_records_button.pack(side=TOP)

pdf_frame = Frame(right_frame)
pdf_frame.pack(side=LEFT, fill=BOTH)
scrol_y = Scrollbar(pdf_frame, orient=VERTICAL)
pdf = Text(pdf_frame, yscrollcommand=scrol_y.set, bg="grey")
scrol_y.pack(side=RIGHT, fill=Y)
scrol_y.config(command=pdf.yview)
pdf.pack(fill=BOTH, expand=1)

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

bottom_frame = Frame(root)
bottom_frame.pack(side=BOTTOM, fill=BOTH, expand=TRUE)

folder_manager_button = Button(text='Folders', command=ref_manager.open_folder_window)
folder_manager_button.pack(in_=bottom_frame, fill=Y, ipady=5, ipadx=5, side=LEFT)

get_text_button = Button(text='Dictionary', command=ref_manager.open_dictionary_window)
get_text_button.pack(in_=bottom_frame, fill=Y, ipady=5, ipadx=5, side=LEFT)

save_button = Button(text='Save', command=ref_manager.save_data)
save_button.pack(in_=bottom_frame, fill=Y, ipady=5, ipadx=5, side=LEFT)

ref_manager.load_data()

root.mainloop()

