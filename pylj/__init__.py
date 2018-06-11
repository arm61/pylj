import webbrowser

def __cite__():
    webbrowser.open('https://zenodo.org/badge/latestdoi/119863480')

def __version__():
    major=0
    minor=0
    micro=22
    print('pylj-{:d}.{:d}.{:d}'.format(major, minor, micro))