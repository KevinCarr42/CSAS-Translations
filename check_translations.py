import json


def load_translations(json_file="all_translations.json"):
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def is_word_in_translations(word, translations_data=None):
    if translations_data is None:
        translations_data = load_translations()
    
    word_lower = word.lower()
    
    for category, translations in translations_data['translations'].items():
        for term, translation in translations.items():
            if term.lower() == word_lower:
                return True, category, translation
            if translation and translation.lower() == word_lower:
                return True, category, term
    
    return False, None, None


def find_word_translations(word, translations_data=None):
    if translations_data is None:
        translations_data = load_translations()
    
    word_lower = word.lower()
    matches = []
    
    for category, translations in translations_data['translations'].items():
        for term, translation in translations.items():
            if term.lower() == word_lower:
                matches.append({
                    'category': category,
                    'original': term,
                    'translation': translation
                })
    
    return matches


if __name__ == "__main__":
    translations_data = load_translations()
    for word in ["abondance", "augmentation", "environnement"]:
        print(word, is_word_in_translations(word, translations_data))
