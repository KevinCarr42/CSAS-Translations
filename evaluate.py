import random
import os
from datasets import load_dataset
from datetime import datetime
import json
import csv
from sentence_transformers import SentenceTransformer
from translate import OpusTranslationModel, M2M100TranslationModel, MBART50TranslationModel, TranslationManager

language_codes = {
    "en": "English",
    "fr": "French",
}


def sample_data(path, n_samples=10, source_lang=None,
                use_eval_split=False, val_ratio=0.05, split_seed=42):
    if use_eval_split:
        ds = load_dataset("json", data_files=path, split="train")
        if source_lang:
            ds = ds.filter(lambda x: x.get("source_lang") == source_lang, load_from_cache_file=False)
        eval_ds = ds.train_test_split(test_size=val_ratio, seed=split_seed)["test"]
        k = len(eval_ds) if n_samples is None else min(n_samples, len(eval_ds))
        if k < len(eval_ds):
            idx = random.sample(range(len(eval_ds)), k)
            return [eval_ds[i] for i in idx]
        return [eval_ds[i] for i in range(len(eval_ds))]
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f
                    if not source_lang or json.loads(line).get("source_lang") == source_lang]
        k = len(data) if n_samples is None else min(n_samples, len(data))
        return random.sample(data, k) if k < len(data) else data


def test_translations_with_loaded_models(translation_manager, dataset, name_suffix=None, use_find_and_replace=True):
    n_samples = len(dataset)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    INDENT = 70
    
    if name_suffix:
        csv_path = f"translation_results/{ts}_translation_comparison_{name_suffix}.csv"
        errors_path = f"translation_results/{ts}_translation_errors_{name_suffix}.json"
    else:
        csv_path = f"translation_results/{ts}_translation_comparison.csv"
        errors_path = f"translation_results/{ts}_translation_errors.json"
    
    csv_data = []
    
    print(f"\nRunning test: {name_suffix if name_suffix else 'default'}")
    print(f"Total samples: {n_samples}")
    print("-" * 80)
    
    for i, d in enumerate(dataset, start=1):
        source = d.get("source")
        target = d.get("target")
        source_lang = d.get("source_lang")
        other_lang = "en" if source_lang == "fr" else "fr"
        
        print(
            f"\n[sample {i}/{n_samples}] {language_codes[source_lang]}"
            f"\n{f'text in ({language_codes[source_lang]}):':<{INDENT}}{source}"
            f"\n{f'text out ({language_codes[other_lang]}), expected:':<{INDENT}}{target}"
        )
        
        translation_result = translation_manager.translate_with_all_models(
            text=source,
            source_lang=source_lang,
            target_lang=other_lang,
            use_find_replace=use_find_and_replace,
            idx=i,
            target_text=target,
        )
        
        for model_name, result in translation_result.items():
            csv_entry = {
                'source': source,
                'target': target,
                'source_lang': source_lang,
                'other_lang': other_lang,
                'translator_name': model_name,
                'translated_text': result.get("translated_text", "[TRANSLATION FAILED]"),
                'cosine_similarity_original_translation': result.get("similarity_of_original_translation"),
                'cosine_similarity_vs_source': result.get("similarity_vs_source"),
                'cosine_similarity_vs_target': result.get("similarity_vs_target"),
            }
            csv_data.append(csv_entry)
            
            print(f"{f'text out ({language_codes[other_lang]}), predicted with {model_name}:':<{INDENT}}"
                  f"{result.get('translated_text', '[FAILED]')}")
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'source', 'target', 'source_lang', 'other_lang', 'translator_name', 'translated_text',
            'cosine_similarity_original_translation', 'cosine_similarity_vs_source', 'cosine_similarity_vs_target'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)
    
    error_summary = translation_manager.get_error_summary()
    if error_summary["extra_token_errors"] > 0 or error_summary["find_replace_errors"] > 0:
        with open(errors_path, "w", encoding="utf-8") as f:
            json.dump(error_summary, f, ensure_ascii=False, indent=2)
    
    print(f"\nCompleted test: {name_suffix if name_suffix else 'default'}")
    print(f"Results saved to: {csv_path}")
    print(f"Total samples processed: {n_samples}")
    print(f"Total CSV entries written: {len(csv_data)}")
    
    if error_summary["extra_token_errors"] > 0:
        print(f"Extra token errors: {error_summary['extra_token_errors']}")
    if error_summary["find_replace_errors"] > 0:
        print(f"Find-replace errors: {error_summary['find_replace_errors']}")
    if error_summary["extra_token_errors"] > 0 or error_summary["find_replace_errors"] > 0:
        print(f"Error details saved to: {errors_path}")
    else:
        print("No Errors! Error log not saved.")


if __name__ == "__main__":
    random.seed(42)
    os.makedirs("translation_results", exist_ok=True)
    
    training_data = "../Data/training_data.jsonl"
    testing_data = "../Data/testing_data.jsonl"
    merged_model_folder = "../Data/merged/"
    
    all_models = {
        "opus_mt_base": {
            "cls": OpusTranslationModel,
            "params": {
                "base_model_id": "Helsinki-NLP/opus-mt-tc-big-en-fr",
                "model_type": "seq2seq",
            }
        },
        "opus_mt_finetuned": {
            "cls": OpusTranslationModel,
            "params": {
                "base_model_id": "Helsinki-NLP/opus-mt-tc-big-en-fr",
                "model_type": "seq2seq",
                "merged_model_path_en_fr": f"{merged_model_folder}opus_mt_en_fr",
                "merged_model_path_fr_en": f"{merged_model_folder}opus_mt_fr_en",
            }
        },
        
        "m2m100_418m_base": {
            "cls": M2M100TranslationModel,
            "params": {
                "base_model_id": "facebook/m2m100_418M",
                "model_type": "seq2seq",
            }
        },
        "m2m100_418m_finetuned": {
            "cls": M2M100TranslationModel,
            "params": {
                "base_model_id": "facebook/m2m100_418M",
                "model_type": "seq2seq",
                "merged_model_path": f"{merged_model_folder}m2m100_418m",
            }
        },
        
        "mbart50_mmt_base": {
            "cls": MBART50TranslationModel,
            "params": {
                "base_model_id": "facebook/mbart-large-50-many-to-many-mmt",
                "model_type": "seq2seq",
            }
        },
        "mbart50_mmt_finetuned": {
            "cls": MBART50TranslationModel,
            "params": {
                "base_model_id": "facebook/mbart-large-50-many-to-many-mmt",
                "model_type": "seq2seq",
                "merged_model_path_en_fr": f"{merged_model_folder}mbart50_mmt_fr",
                "merged_model_path_fr_en": f"{merged_model_folder}mbart50_mmt_en",
            }
        },
    }
    
    print("\nLoading embedder...")
    embedder = SentenceTransformer('sentence-transformers/LaBSE')
    print("Embedder loaded successfully!\n")
    
    n_tests = 1000
    print(f"Sampling {n_tests} examples from datasets...")
    sampled_testing_data = sample_data(testing_data, n_tests, use_eval_split=False)
    sampled_training_data = sample_data(training_data, n_tests, use_eval_split=True)
    
    translation_manager = TranslationManager(all_models, embedder)
    print("Loading models...")
    translation_manager.load_models()
    
    test_translations_with_loaded_models(translation_manager, sampled_testing_data, "test", True)
    test_translations_with_loaded_models(translation_manager, sampled_training_data, "train", True)
    test_translations_with_loaded_models(translation_manager, sampled_testing_data, "test_no_find_and_replace", False)
    test_translations_with_loaded_models(translation_manager, sampled_training_data, "train_no_find_and_replace", False)
    
    print("\nAll tests completed!")
