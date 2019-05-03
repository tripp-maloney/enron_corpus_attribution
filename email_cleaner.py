import re

email_list = "all_emails_dirty.tsv"


def forward_scrape(line):
    scraped = re.sub(r"---------------------- Forwarded by .*", "", line)
    return scraped


def orig_scrape(line):
    scraped = re.sub(r"-----Original Message-----.*", "", line)
    return scraped


def from_scrape(line):
    scraped = re.sub(r" {3}From:", "", line)
    return scraped


def spaces_scrape(line):
    scraped = re.sub(r" {5}.*", "", line)
    return scraped


def scrape(line):
    round1 = forward_scrape(line)
    round2 = orig_scrape(round1)
    scraped = spaces_scrape(round2)
    return scraped


def field_split(line):
    fields = line.split("\t")
    return fields


def delete_empty_lines(lines):
    no_empties = [line for line in lines if field_split(line)[2] != ""]
    return no_empties


print("Scraping extraneous text from emails...\n")
with open(email_list, "r", encoding='utf-8') as f:
    emails = f.readlines()
    s_emails = [email.strip() for email in emails]

    scraped_emails = [scrape(email) for email in s_emails]
    empties_deleted = delete_empty_lines(scraped_emails)
print("Done!\n\nWriting clean emails to new doc...")
with open("all_emails_dbl_clean.tsv", "w", encoding='utf-8') as out:
    for i in range(len(empties_deleted)):
        out.write(empties_deleted[i]+"\n")
        if i % 1000 == 0:
            print(f"{str(i)} emails written")



