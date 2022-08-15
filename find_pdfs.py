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


def rgbtohex(color):
	r = int(color[0] * 255)
	g = int(color[1] * 255)
	b = int(color[2] * 255)
	return f'#{r:02x}{g:02x}{b:02x}'


def get_items(folder):
	items = [os.path.join(folder, i) for i in os.listdir(folder) if not i.startswith('.')]
	folders = [item for item in items if os.path.isdir(item)]
	pdfs = [item for item in items if item.lower().endswith('.pdf')]
	return folders, pdfs


def check_not_ignore(folder, ignore_folders):
	for f in ignore_folders:
		if f in folder:
			return False
	return True


def get_pdfs(folder, ignore):
	sub_folders, all_pdfs = get_items(folder)
	for sub_folder in sub_folders:
		if check_not_ignore(sub_folder, ignore):
			all_pdfs.extend(get_pdfs(sub_folder, ignore))
	return all_pdfs


def organize_pdfs(pdfs):
	folders = dict()
	for pdf in pdfs:
		folder = os.path.dirname(pdf)
		if folder in folders.keys():
			folders[folder].append(os.path.basename(pdf))
		else:
			folders[folder] = [os.path.basename(pdf)]
	return folders


def get_selected_item():
	item_iid = tree_view.selection()[0]
	parent_iid = tree_view.parent(item_iid)
	parent_value = tree_view.item(parent_iid)['values']
	if parent_value == '':
		# we have selected a folder, do nothing
		folder_path = item_value = tree_view.item(item_iid)['values'][0]
		return folder_path, 'folder'
	else:
		folder = parent_value[0]
		item_value = tree_view.item(item_iid)['values']
		filename = item_value[0].replace('\t', '')
		pdf_path = os.path.join(folder, filename)
		return pdf_path, 'pdf'


def open_path(event):
	path, kind = get_selected_item()
	if kind == 'pdf':
		opener = "open" if sys.platform == "darwin" else "xdg-open"
		subprocess.call([opener, path])


def keydown(e):
	if e.char == 'm':
		path, kind = get_selected_item()
		if kind == 'pdf':
			new_path = os.path.join(target_folder, os.path.basename(path))
			if os.path.exists(new_path):
				print('A file already exists at "{}"'.format(new_path))
			try:
				os.rename(path, new_path)
			except Exception as e:
				print(e)


def select_path(event):
	item_iid = tree_view.selection()[0]
	parent_iid = tree_view.parent(item_iid)
	parent_value = tree_view.item(parent_iid)['values']
	if parent_value == '':
		# we have selected a folder, do nothing
		pass
	else:
		folder = parent_value[0]
		item_value = tree_view.item(item_iid)['values']
		filename = item_value[0].replace('\t', '')
		pdf_path = os.path.join(folder, filename)
		# print(pdf_path)
		pdf_text.delete('1.0', END)
		if pdf_to_image[pdf_path] is not None:
			pdf_id[0] = pdf_text.image_create(END, image=pdf_to_image[pdf_path])


top_level_folders = [
	'/Users/raymondbaranski/Downloads',
	'/Users/raymondbaranski/Documents',
	'/Users/raymondbaranski/Desktop'
]
# top_level_folders = [
# 	'/Users/raymondbaranski/Downloads'
# ]
ignore_folders = [
	'/Users/raymondbaranski/Documents/My EndNote Library.Data',
	'/Users/raymondbaranski/Documents/My EndNote Library-Converted.Data'
]
target_folder = '/Users/raymondbaranski/Literature/target_folder'

pdfs = list()
for folder in top_level_folders:
	pdfs.extend(get_pdfs(folder, ignore_folders))

tree = organize_pdfs(pdfs)

# print(pdfs)

root = tk.Tk()
root.configure(bg=rgbtohex((.5, .5, .5)))
canvas = tk.Canvas(root, bg='black', width=1600, height=900)
canvas.pack()

style = ttk.Style()
style.configure("Mystyle.Treeview",  indent=15, bd=10)

tree_view = ttk.Treeview(canvas, column=('w1'), height=50, selectmode='browse', style="Mystyle.Treeview")
tree_view.column('# 1', anchor=W, stretch=NO, minwidth=250, width=800)
tree_view.heading('# 1', text='PDF')
tree_view.pack(side=LEFT)

pdf_frame = Frame(canvas)
pdf_frame.pack(side=LEFT, fill=BOTH)

pdf_text = Text(pdf_frame, bg="grey")
pdf_text.pack(fill=BOTH, expand=1)
pdf_id = [0]

pdf_to_image = dict()
factor = 550 / 800
for i in range(len(pdfs)):
	print('\r{}/{}\t\t\t'.format(i, len(pdfs)), end='')
	pdf_path = pdfs[i]
	try:
		pages = convert_from_path(pdf_path, size=(800 * factor, 1000 * factor), last_page=1)
		pdf_to_image[pdf_path] = ImageTk.PhotoImage(pages[0])
	except:
		pdf_to_image[pdf_path] = None
print('')


print('\nPDFS:')
i = 1
for folder in tree.keys():
	tree_view.insert('', END, iid=i, values=(folder,))
	# print('{}:'.format(folder))
	for pdf in tree[folder]:
		tree_view.insert(i, END, values=('\t{}'.format(pdf),))
		# print('\t{}'.format(pdf))
	i += 1


# for pdf in pdfs:
# 	print('\t{}'.format(pdf))

tree_view.bind('<<TreeviewSelect>>', select_path)
tree_view.bind('<Return>', open_path)
tree_view.bind('<KeyPress>', keydown)
root.mainloop()