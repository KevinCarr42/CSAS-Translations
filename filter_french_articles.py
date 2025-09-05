import json
from check_translations import is_word_in_translations, load_translations


def filter_french_articles(input_file):
    french_articles = ["La ", "Le ", "L'", "Les "]
    kept_rows = []
    total_count = 0
    
    translations_data = load_translations()
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            total_count += 1
            data = json.loads(line.strip())
            source = data.get('source', '')
            
            article_match = None
            for article in french_articles:
                if source.startswith(article):
                    article_match = article
                    break
            
            if article_match:
                remaining_text = source[len(article_match):].strip()
                words = remaining_text.split()
                
                if len(words) > 0:
                    second_word = words[0].rstrip('.,!?;:')
                    found, _, _ = is_word_in_translations(second_word, translations_data)
                    
                    if found:
                        kept_rows.append(line)
    
    with open(input_file, 'w', encoding='utf-8') as outfile:
        for line in kept_rows:
            outfile.write(line)
    
    kept_count = len(kept_rows)
    print(f"Filtered {input_file}: kept {kept_count}/{total_count} rows")


if __name__ == "__main__":
    filter_french_articles("fake_testing_data.jsonl")
