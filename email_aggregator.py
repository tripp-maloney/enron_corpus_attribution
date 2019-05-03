import os
import re
import nltk

def list_files(dir):
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            r.append(os.path.join(root, name))
    return r

r = list_files("maildir2")

print("fetching emails")
with open("all_emails_dirty.tsv", "a", encoding='utf-8') as out:
    id = 1
    duds = 0
    for file in r:
        with open(file, 'r', encoding='utf-8') as f:
            try:
                lines = f.readlines()
            except UnicodeDecodeError:
                duds +=1
                continue
            slines = [line.strip() for line in lines]
            for i in range(len(lines)):
                last_meta = re.search("X-FileName:", lines[i])
                if last_meta is not None:
                    meta_stripped = slines[i+1:]
                    break
            # relevant_lines =
            body = " ".join(meta_stripped)
            body = body.strip()
            author = os.path.basename(os.path.dirname(os.path.dirname(file)))
            out.write(f"{id:06d}\t" + author + "\t" + body+"\n")
            id += 1
            print(f"{id} of {len(r)}")

print(f"Finished with {duds} unreadable files.")

