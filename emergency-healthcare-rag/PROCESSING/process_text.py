# %%
import os
import json

# %%
# Get all folders in data/topics
folders = [f for f in os.listdir('data/topics') if os.path.isdir(os.path.join('data/topics', f))]


# %%
def format_md(text):
    # Remove everything before the first "## Continuing Education Activity" 
    if "## Continuing Education Activity" in text:
        text = text.split("## Continuing Education Activity", 1)[1]
    # Remove everything before the first "#### Affiliations"
    if "#### Affiliations" in text:
        text = text.split("#### Affiliations", 1)[1]
    # Remove everything before the first "#### Authors"
    if "#### Authors" in text:
        text = text.split("#### Authors", 1)[1]
    # Remove everything before the first "## Introduction"
    if "## Introduction" in text:
        text = text.split("## Introduction", 1)[1]
    # remove everything and including after the last "## Review Questions"
    if "## Review Questions" in text:
        text = text.split("## Review Questions", 1)[0]
    # remove everything and including after the last "## References"
    if "## References" in text:
        text = text.split("## References", 1)[0]

    return text.strip()

# %%
for folder in folders:
    os.makedirs(os.path.join('data_txt/topics', folder), exist_ok=True)

    # read md files in each folder
    md_files = [f for f in os.listdir(os.path.join('data/topics', folder)) if f.endswith('.md')]
    for md_file in md_files:
        with open(os.path.join('data/topics', folder, md_file), 'r') as f:
            content = f.read()
            content = format_md(content)
        
        print(len(content.split(' ')), folder, md_file)

        # save content to .txt file
        txt_file = md_file.replace('.md', '.txt')
        with open(os.path.join('data_txt/topics', folder, txt_file), 'w') as f:
            f.write(content)


