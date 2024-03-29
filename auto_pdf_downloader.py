import os
import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter import simpledialog
from pdf2image import convert_from_path
import PIL
from PIL import Image, ImageTk
import sys, subprocess
import webbrowser
import requests


'''
elifesciences sucks

'''

ARXIV = '//arxiv.org'  #
SCIENCEDIRECT = '//www.sciencedirect.com'
FRONTIER = '//www.frontiersin.org'
JOURNALS_PHYS = '//journals.physiology.org'
JOURNALS_PLOS = '//journals.plos.org'
SPRINGER = '//link.springer.com'
OPENREVIEW = '//openreview.net'
BIORXIV = '//www.biorxiv.org'
NATURE = '//www.nature.com'
NIH = '//www.ncbi.nlm.nih.gov'



def write_pdf_to_file(website, pdf_folder, pdf_name):
	if not os.path.exists(os.path.join(pdf_folder, pdf_name)):
		try:
			headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntsu; Linux x86_64; rv:103.0) Gecko/20100101 Firefox/103.0'}
			response = requests.get(website, headers=headers, allow_redirects=True, verify=False)
			content = response.content
			if content[0:4].decode('utf-8') == '%PDF':
				with open(os.path.join(pdf_folder, pdf_name), 'wb') as f:
					f.write(content)
				return True
			else:
				# print(content[0:4].decode('utf-8'))
				return False
		except Exception as e:
			print(e)
			return False
	return True


def select_link(event):
	selection = listbox.curselection()
	listbox.itemconfig(selection[0], bg=rgbtohex((0, 1, 0)))
	link = listbox.get(selection)

	pdf_link = pdf_link_from_website(link)
	if pdf_link is not None:
		if not download_pdf_from_link(pdf_link):
			open_link(pdf_link)


def open_link(link):
	controller = webbrowser.get('Firefox')
	controller.open(link)


def rgbtohex(color):
	r = int(color[0] * 255)
	g = int(color[1] * 255)
	b = int(color[2] * 255)
	return f'#{r:02x}{g:02x}{b:02x}'


def domain_from_website(website):
	website = website.replace('https://', '').replace('http://', '').replace('file:///', '')
	return website.split('/')[0]


def pdf_link_from_website(website):
	if website.endswith('.pdf'):
		return website
	else:
		if ARXIV in website:
			return website.replace('/abs/', '/pdf/') + '.pdf'
		elif SCIENCEDIRECT in website:
			if '/abs/pii/' in website:
				return website
			else:
				return website + '/pdfft?isDTMRedir=true'
		elif JOURNALS_PHYS in website:
			return website.replace('/full/', '/pdf/') + '?download=true'
		elif JOURNALS_PLOS in website:
			return website.replace('?id=', '/file?id=') + '&type=printable'
		elif SPRINGER in website:
			return website.replace('/article/', '/content/pdf/') + '.pdf'
		elif OPENREVIEW in website:
			return website.replace('/forum?', '/pdf?')
		elif BIORXIV in website:
			if website.endswith('.full'):
				return website + '.pdf'
			elif website.endswith('.abstract'):
				return website.replace('.abstract', '.full.pdf')
			else:
				return website + '.full.pdf'
		elif FRONTIER in website:
			return website.replace('/full', '/pdf')
		elif NATURE in website:
			return website + '.pdf'
		elif NIH in website:
			return website + 'pdf'
		else:
			return website


def pdf_name_from_link(website):
	# this is from the pdf_link, NOT the raw link
	if website.endswith('.pdf'):
		return website.split('/')[-1]
	if ARXIV in website:
		return website.split('/')[-1]
	elif SCIENCEDIRECT in website:
		return website.split('/')[-2] + '-main.pdf'
	elif JOURNALS_PHYS in website:
		return website.replace('?download=true', '').split('/')[-1] + '.pdf'
	elif JOURNALS_PLOS in website:
		return website.replace('&type=printable', '').split('/')[-1] + '.pdf'
	elif SPRINGER in website:
		return website.split('/')[-1]
	elif OPENREVIEW in website:
		return 'openreview_' + website.split('/pdf?')[-1] + '.pdf'
	elif BIORXIV in website:
		return website.replace('.full.pdf', '').split('/')[-1] + '.pdf'
	elif FRONTIER in website:
		return website.split('/')[-2] + '.pdf'
	elif NATURE in website:
		return website.split('/')[-1]
	elif NIH in website:
		return website.split('/')[5] + '.pdf'


def download_pdf_from_link(website):
	pdf_name = pdf_name_from_link(website)

	if ARXIV in website:
		return write_pdf_to_file(website, pdf_folder, pdf_name)
	elif SCIENCEDIRECT in website:
		return write_pdf_to_file(website, pdf_folder, pdf_name)
	elif JOURNALS_PHYS in website:
		return write_pdf_to_file(website, pdf_folder, pdf_name)
		# os.rename(os.path.join(downloads_folder, pdf_name), os.path.join(downloads_folder, pdf_name))
	elif JOURNALS_PLOS in website:
		return write_pdf_to_file(website, pdf_folder, pdf_name)
	elif SPRINGER in website:
		return write_pdf_to_file(website, pdf_folder, pdf_name)
	elif OPENREVIEW in website:
		return write_pdf_to_file(website, pdf_folder, pdf_name)
	elif BIORXIV in website:
		return write_pdf_to_file(website, pdf_folder, pdf_name)
	elif FRONTIER in website:
		return write_pdf_to_file(website, pdf_folder, pdf_name)
	elif NATURE in website:
		return write_pdf_to_file(website, pdf_folder, pdf_name)
	elif NIH in website:
		return write_pdf_to_file(website, pdf_folder, pdf_name)
	else:
		if website.endswith('.pdf'):
			return write_pdf_to_file(website, pdf_folder, pdf_name)
		else:
			return False


downloads_folder = '/home/alex/Downloads'
path = '/home/alex/Documents/GitHub/PaperManager/'
filepath = os.path.join(path, 'science_tabs_2.txt')
pdf_folder = os.path.join(path, 'pdfs')

with open(filepath, 'r') as f:
	science_links = f.readlines()
science_links = [link.strip() for link in science_links]

bad_links = list()
n_links = len(science_links)
i = 0
for link in science_links:
	i += 1
	print('\r{}/{}\t\t\t'.format(i, n_links), end='')
	pdf_link = pdf_link_from_website(link)
	# print(pdf_link)
	if pdf_link is not None:
		if not download_pdf_from_link(pdf_link):
			bad_links.append(link)
	else:
		bad_links.append(link)
print('\nDone')

with open(os.path.join(path, 'science_tabs_3.txt'), 'w') as f:
	for link in bad_links:
		f.write('{}\n'.format(link))

# root = tk.Tk()
# root.configure(bg=rgbtohex((.5, .5, .5)))
# canvas = tk.Canvas(root, bg='black', width=1600, height=900)
# canvas.pack()
#
# listbox = Listbox(canvas, width=150, height=50)
# listbox.pack()
#
# for link in science_links:
# 	listbox.insert(END, link)
#
# listbox.bind('<Return>', select_link)
#
# root.mainloop()