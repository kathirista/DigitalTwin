import os
import glob
import json
from bs4 import BeautifulSoup

def process_html_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract title
    title_tag = soup.find('h1', class_='p-name')
    if not title_tag:
        title_tag = soup.find('title')
    title_text = title_tag.get_text(strip=True) if title_tag else ""
    
    # Extract canonical link (URL)
    url_tag = soup.find('a', class_='p-canonical')
    url = url_tag['href'] if url_tag and url_tag.has_attr('href') else ""
    
    # Extract date
    time_tag = soup.find('time', class_='dt-published')
    date = time_tag['datetime'] if time_tag and time_tag.has_attr('datetime') else ""
    
    # Extract content
    content_tag = soup.find('section', class_='e-content')
    if not content_tag:
        content_tag = soup.find('article')
    
    # Remove script and style elements
    if content_tag:
        for script_or_style in content_tag(["script", "style"]):
            script_or_style.extract()
        content = content_tag.get_text(separator=' ', strip=True)
    else:
        for script_or_style in soup.body(["script", "style"]):
            script_or_style.extract()
        content = soup.body.get_text(separator=' ', strip=True) if soup.body else ""
        
    return {
        "title": title_text,
        "date": date,
        "url": url,
        "filename": os.path.basename(filepath),
        "content": content
    }

def main():
    directory = '/home/sruthi_korlakunta/MediumPosts'
    json_path = os.path.join(directory, 'knowledge_base.json')
    jsonl_path = os.path.join(directory, 'knowledge_base.jsonl')
    
    files = glob.glob(os.path.join(directory, '*.html'))
    
    data = []
    for filepath in files:
        try:
            item = process_html_file(filepath)
            data.append(item)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            
    # Write JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        
    # Write JSONL
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"Successfully processed {len(data)} files.")
    print(f"Database saved to {json_path} and {jsonl_path}")

if __name__ == '__main__':
    main()