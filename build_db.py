import os
import glob
import json
from bs4 import BeautifulSoup

base_dir = "/home/sruthi_korlakunta/MediumPosts"
html_files = glob.glob(os.path.join(base_dir, "*.html"))

data = []

for filepath in html_files:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        soup = BeautifulSoup(content, 'html.parser')
        
        # We need to exclude comments. The user specified "these are files where content is a maximum of 5 lines."
        # Let's get the content blocks in the body.
        body = soup.find('section', {'data-field': 'body'})
        if not body:
            # Maybe the structure is slightly different, let's grab all text from body if section is not found
            # but usually it's there.
            body = soup.find('article') or soup.find('body')
        
        blocks = []
        if body:
            for tag in body.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'blockquote', 'pre']):
                text = tag.get_text(strip=True)
                if text:
                    blocks.append(text)
        
        # Use simple line count of block content to decide if it's <= 5 lines
        if len(blocks) <= 5:
            continue
            
        full_text = "\n\n".join(blocks)
        
        # Extract title
        title_tag = soup.title
        title = title_tag.get_text(strip=True) if title_tag else os.path.basename(filepath)
        
        # Try to find a better title from header h1
        header_h1 = soup.find('header')
        if header_h1:
            h1 = header_h1.find('h1')
            if h1:
                title = h1.get_text(strip=True)
                
        # Extract publish date
        time_tag = soup.find('time', class_='dt-published')
        publish_date = time_tag['datetime'] if time_tag and 'datetime' in time_tag.attrs else ""
        if not publish_date and time_tag:
             publish_date = time_tag.get_text(strip=True)
             
        # Extract Canonical URL
        canonical_tag = soup.find('a', class_='p-canonical')
        url = canonical_tag['href'] if canonical_tag and 'href' in canonical_tag.attrs else ""
        
        # Create record
        record = {
            "id": os.path.basename(filepath),
            "title": title,
            "publish_date": publish_date,
            "url": url,
            "content": full_text
        }
        
        data.append(record)
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

# Save to json
out_json = os.path.join(base_dir, "database.json")
with open(out_json, "w", encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

# Save to jsonl
out_jsonl = os.path.join(base_dir, "database.jsonl")
with open(out_jsonl, "w", encoding='utf-8') as f:
    for entry in data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Processed {len(html_files)} files. Kept {len(data)} articles (excluding comments). Saved to database.json and database.jsonl")