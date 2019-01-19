counter = 0
content = ''

with open('../data/parse-wiktionary/wiki.xml') as infile:
    for line in infile:
        counter += 1
        content += line
        if counter > 1000000:
            break

pages = content.split('</page>\n  <page>')
for page in pages[:10]:
    title_start = page.find('<title>')
    title_end = page.find('</title>')
    title = page[title_start + 7:title_end]

    def_start = page.find('==English==')
    def_end = page.find('----')
    definition = page[def_start:def_end]

    types = ['Noun', 'Adjective', 'Adverb', 'Verb']
    word_start = [definition.find(f'==={type}===') for type in types]
    word_start = [ws for ws in word_start if ws != -1]
    if len(word_start) == 0:
        word_start = word_end = 0
    else:
        word_start = min(word_start)
        word_end = definition[word_start:].find('====')

    print(title)
    print(definition[word_start:word_start+word_end])
    print('\n\n\n')
