authors = {0: 'Andrew R. McCluskey', 1:'Benjamin J. Morgan', 2:'Karen J. Edler', 3:'Stephen C. Parker'}

citation = {'released':'2018-05-15', 'name':'pylj', 'version':'0.0.6a', 'doi':'10.5281/zenodo.1212792',
            'note':'Thank you for using pylj. If you use this code in a teaching laboratory or a publication we would greatly appreciate if you would use the following citation.'}

def __cite__():
    author_list = ''
    for i in range(0, len(authors)):
        if i > 0:
            author_list += ', '
        author_list += authors[i]
    citation_print = '{}\n{} ({}). {}, version {}. Released: {}, DOI: {}'. format(citation['note'],
                                                                                author_list,
                                                                                citation['released'][:4],
                                                                                citation['name'],
                                                                                citation['version'],
                                                                                citation['released'],
                                                                                citation['doi'])
    print(citation_print)