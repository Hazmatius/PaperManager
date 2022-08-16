import os


def domain_from_website(website):
	website = website.replace('https://', '').replace('http://', '').replace('file:///', '')
	return website.split('/')[0]


path = '/home/alex/Documents/GitHub/PaperManager/'
filepath = os.path.join(path, 'onetabs.txt')

with open(filepath, 'r') as f:
	lines = f.readlines()

items = dict()
for line in lines:
	res = line.replace('\n', '').split(' | ')
	if len(res) != 0:
		website = res[0]
		contents = res[1:]
		items[website] = contents

websites = sorted(items.keys())

domains = dict()
for website in websites:
	domain = domain_from_website(website)
	if domain not in domains.keys():
		domains[domain] = dict()
	domains[domain][website] = items[website]

n_items_in_domain = list()
for domain in sorted(domains.keys()):
	n_items_in_domain.append((domain, len(domains[domain].keys())))

science_domains = [
	'arxiv.org',
	'scholar.google.com',
	'www.sciencedirect.com',
	'www.frontiersin.org',
	'www.nature.com',
	'www.researchgate.net',
	'pubmed.ncbi.nlm.nih.gov',
	'www.ncbi.nlm.nih.gov',
	'link.springer.com',
	'ocw.mit.edu',
	'ieeexplore.ieee.org',
	'proceedings.neurips.cc',
	'journals.plos.org',
	'www.biorxiv.org',
	'www.semanticscholar.org',
	'www.pnas.org',
	'elifesciences.org',
	'openreview.net',
	'citeseerx.ist.psu.edu',
	'www.tandfonline.com',
	'onlinelibrary.wiley.com',
	'www.cell.com',
	'proceedings.mlr.press',
	'reader.elsevier.com',
	'ctan.math.illinois.edu',
	'journals.physiology.org',
	'www.science.org',
	'neuro2022.jnss.org'
]

science_links = list()
other_links = list()

for domain in domains.keys():
	if domain in science_domains:
		for website in domains[domain]:
			science_links.append(website)
	else:
		for website in domains[domain]:
			other_links.append(website)

with open(os.path.join(path, 'science_tabs.txt'), 'w') as f:
	for link in science_links:
		f.write('{}\n'.format(link))

with open(os.path.join(path, 'other_tabs.txt'), 'w') as f:
	for link in other_links:
		f.write('{}\n'.format(link))
