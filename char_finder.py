import latex_to_unicode


converter = latex_to_unicode.AccentConverter()

string = converter.decode_Tex_Accents(r'\textquotedblleft')
print(string)
# for i in range(32, 1024):
# 	print('  {} : {}  '.format(str(i).zfill(4), chr(i)), end='')
# 	if i % 10 == 0:
# 		print('\n\n\n', end='')


