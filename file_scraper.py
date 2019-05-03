import os

def iterate(authdir, new_file, id):
    id_num = id
    for subdir, dirs, files in os.walk(authdir):
        for file in files:
            with open(file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                relevant_lines = lines[16:]
                # The first 16 lines of all of the Enron email docs are metadata and whitespace
                body = " ".join(relevant_lines)
                author = authdir
                with open(new_file, "a", encoding='utf-8') as new:
                    new.append(f"{id:08d}\t"+author+"\t"+body)
                    id_num += 1


def main(directory):
    """
    :param directory: Enron corpus maildir folder - contains emails that will be added to THE BIG DOC. 
    :return: A master .tsv doc containing email ID, email author, and email text on each line.
    """

    id = 0
    rootdir = "maildir"
    new_file = "enron_doc.tsv"
    for subdir, dirs, files in os.walk(rootdir):
        for dir in dirs:
            # iterate(dir, new_file, id)
            for subdir2, dirs2, files2 in os.walk(dir):
                for file in files2:
                    with open(file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        relevant_lines = lines[16:]
                        # The first 16 lines of all of the Enron email docs are metadata and whitespace
                        body = " ".join(relevant_lines)
                        author = dir
                        with open(new_file, "a", encoding='utf-8') as new:
                            new.append(f"{id:08d}\t" + author + "\t" + body)
                            id += 1

main("maildir")
