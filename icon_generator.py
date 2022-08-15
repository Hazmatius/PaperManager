import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
import os


def get_mask_from_img(img):
	np_array = np.array(img)
	color_mask = 255 - np_array[:, :, 0:3]
	alpha_mask = np_array[:, :, 3:4]
	return color_mask, alpha_mask


def colorize_img(img, color):
	color_mask, alpha_mask = get_mask_from_img(img)
	color_mask = color * color_mask
	np_array = np.concatenate([color_mask, alpha_mask], axis=2)
	return Image.fromarray(np.uint8(np_array))


def gen_icon(icon_name, source, outline_color, fill_color, shade_color):
	resources_folder = '/Users/raymondbaranski/GitHub/PaperManager/resources'
	outline = Image.open(os.path.join(resources_folder, source, 'outline.png'))
	outline = colorize_img(outline, outline_color)
	fill = Image.open(os.path.join(resources_folder, source, 'fill.png'))
	fill = colorize_img(fill, fill_color)
	shade = Image.open(os.path.join(resources_folder, source, 'shade.png'))
	shade = colorize_img(shade, shade_color)

	final_image = fill
	final_image = Image.alpha_composite(final_image, outline)
	final_image = Image.alpha_composite(final_image, shade)

	final_image.save(os.path.join(resources_folder, '{}.png'.format(icon_name)), format='png')


gen_icon('confirmed', 'large_ball', [0, 0, 1], [0, .5, .5], [0, .7, 1])
gen_icon('match', 'large_ball', [0, .5, 0], [0, .75, 0], [0, .9, 0])
gen_icon('bad_records', 'large_ball', [.5, 0, 0], [.75, 0, 0], [1, 0, 0])
gen_icon('no_status', 'large_ball', [.5, .5, .5], [.75, .75, .75], [1, 1, 1])

gen_icon('near_match', 'exclamation', [0, .4, 0], [0, .8, 0], [0, 0, 0])
gen_icon('bad_bibtex', 'exclamation', [.5, 0, 0], [1, 1, 0], [0, 0, 0])
gen_icon('no_records', 'exclamation', [.5, 0, 0], [1, 0, 0], [0, 0, 0])

# resources_folder = '/Users/raymondbaranski/GitHub/PaperManager/resources'
# outline = Image.open(os.path.join(resources_folder, 'large_ball', 'outline.png'))
# outline = colorize_img(outline, [1, 0, 0])
#
# plt.imshow(outline)
# plt.show()