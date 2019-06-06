import re

email_list = "all_emails_dirty.tsv"


def xml_scrape(line):
    scraped = re.sub(r"<.*>", "", line)
    return scraped


def dash_scrape(line):
    scraped = re.sub(r"----.*", "", line)
    return scraped


def from_scrape(line):
    scraped = re.sub(r"From:.*", "", line)
    return scraped


def spaces_scrape(line):
    scraped = re.sub(r" {5}.*", "", line)
    return scraped


def outlook_scrape(line):
    scraped = re.sub(r"Outlook Migration Team@ENRON.*", "", line)
    return scraped


def enron_sign_scrape(line):
    scraped = re.sub(r" {4}Enron North America Corp.*", "", line)
    return scraped


def to_scrape(line):
    scraped = re.sub(r"To:.*", "", line)
    return scraped


def scrape(line):
    round1 = xml_scrape(line)
    round2 = dash_scrape(round1)
    round3 = from_scrape(round2)
    round4 = outlook_scrape(round3)
    round5 = enron_sign_scrape(round4)
    round6 = to_scrape(round5)
    scraped = spaces_scrape(round6)
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



