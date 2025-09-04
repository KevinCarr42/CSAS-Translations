import json
import re
import subprocess
import sys
import spacy


def ensure_spacy_model(model_name="en_core_web_sm"):
    try:
        nlp_out = spacy.load(model_name)
        return nlp_out
    except OSError:
        print(f"Model '{model_name}' not found. Downloading now...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
        print(f"Model '{model_name}' downloaded successfully!")
        return spacy.load(model_name)


nlp = ensure_spacy_model("en_core_web_sm")


def load_translations(json_file="all_translations.json"):
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_search_patterns(translations_data):
    patterns = {}
    
    for category, translations in translations_data['translations'].items():
        # Sort by length (longest first) to match longer phrases first
        sorted_terms = sorted(translations.keys(), key=len, reverse=True)
        patterns[category] = sorted_terms
    
    return patterns


def preserve_capitalization(original_text, replacement_text):
    if not original_text or not replacement_text:
        return replacement_text
    
    if original_text.isupper():
        return replacement_text.upper()
    elif original_text.islower():
        return replacement_text.lower()
    elif original_text[0].isupper():
        return replacement_text.capitalize()
    else:
        return replacement_text


def detect_places_with_nlp(text):
    places = []
    
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ['GPE', 'LOC', 'FAC']:  # GPE=Geopolitical, LOC=Location, FAC=Facility
            places.append((ent.start_char, ent.end_char, ent.text))
    
    return places


def preprocess_for_translation(text, translations_file="all_translations.json"):
    translations_data = load_translations(translations_file)
    patterns = create_search_patterns(translations_data)
    
    processed_text = text
    token_mapping = {}
    token_counters = {
        'nomenclature': 0,
        'taxon': 0,
        'site': 0,
        'acronym': 0,
        'nlp_places': 0
    }
    
    # Step 1: First detect NLP places and tokenize them
    nlp_places = detect_places_with_nlp(processed_text)
    for start, end, place_text in reversed(nlp_places):  # Reverse to maintain indices
        token_counters['nlp_places'] += 1
        token = f"SITE{token_counters['nlp_places']:04d}"
        
        # Check if this place has a known translation
        place_translation = None
        for place_key, translation in translations_data['translations']['site'].items():
            if place_key.lower() == place_text.lower():
                place_translation = translation
                break
        
        # Store mapping
        token_mapping[token] = {
            'original_text': place_text,
            'category': 'nlp_places',
            'translation': place_translation,
            'should_translate': place_translation is not None
        }
        
        # Replace in text
        processed_text = processed_text[:start] + token + processed_text[end:]
    
    # Step 2: Process dictionary-based terms (excluding site since NLP handles those)
    for category, terms in patterns.items():
        if category == 'site':
            continue  # Skip - handled by NLP above
        
        category_short = category.split('_')[0].upper()
        
        for term in terms:
            # Case-insensitive search for the term as whole words/phrases
            # Use word boundaries for single words, but allow phrase matching
            if ' ' in term:
                # Multi-word phrase - match exactly
                pattern = re.compile(re.escape(term), re.IGNORECASE)
            else:
                # Single word - use word boundaries to avoid partial matches
                pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
            
            matches = list(pattern.finditer(processed_text))
            
            for match in reversed(matches):  # Reverse to maintain indices
                original_text = match.group()
                start, end = match.span()
                
                # Create token
                token_counters[category] += 1
                token = f"{category_short}{token_counters[category]:04d}"
                
                # Get the correct translation key (use original case from translations)
                translation_key = None
                for original_key in translations_data['translations'][category].keys():
                    if original_key.lower() == term.lower():
                        translation_key = original_key
                        break
                
                # Store mapping
                token_mapping[token] = {
                    'original_text': original_text,
                    'category': category,
                    'translation': translations_data['translations'][category].get(translation_key, None) if translation_key else None,
                    'should_translate': True
                }
                
                # Replace in text
                processed_text = processed_text[:start] + token + processed_text[end:]
    
    return processed_text, token_mapping


def postprocess_translation(translated_text, token_mapping):
    result_text = translated_text
    
    for token in token_mapping.keys():
        if token in result_text:
            mapping = token_mapping[token]
            
            if mapping['should_translate'] and mapping['translation'] and mapping['translation'] != 'None':
                replacement = preserve_capitalization(mapping['original_text'], mapping['translation'])
            else:
                replacement = mapping['original_text']
            
            result_text = result_text.replace(token, replacement)
    
    return result_text


def get_translation_statistics(token_mapping):
    stats = {}
    for token, mapping in token_mapping.items():
        category = mapping['category']
        if category not in stats:
            stats[category] = 0
        stats[category] += 1
    
    return stats
