import os

def list_files(dir):
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            r.append(os.path.join(root, name))
    return r

r = list_files("maildir")

print("fetching emails")
with open("all_enron_corpus_emails.tsv", "a", encoding='utf-7') as out:
    id = 1
    authors = []
    elements = []
    for file in r:
        author = os.path.basename(os.path.dirname(os.path.dirname(file)))
        authors.append(author)
        if author not in elements:
            elements.append(author)
    for element in elements:
        print(f"{element} has {str(authors.count(element))} emails.")