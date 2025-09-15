import random
import os
from datasets import load_dataset
from datetime import datetime
import json
import csv
from sentence_transformers import SentenceTransformer
from translate import (NLLBTranslationModel, OpusTranslationModel, M2M100TranslationModel,
                       MBART50TranslationModel, TranslationManager)

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


def test_translations_with_loaded_models(translation_manager, dataset, name_suffix=None, bypass_rules=True):
    n_samples = len(dataset)
    ts = datetime.now().strftime("%Y%m%d-%H%M")
    INDENT = 70
    
    if name_suffix:
        csv_path = f"translation_results/translation_comparison_{name_suffix}_{ts}.csv"
        errors_path = f"translation_results/translation_errors_{name_suffix}_{ts}.json"
    else:
        csv_path = f"translation_results/translation_comparison_{ts}.csv"
        errors_path = f"translation_results/translation_errors_{ts}.json"
    
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
        
        translation_result = translation_manager.translate_with_best_model(
            text=source,
            target_text=target,
            source_lang=source_lang,
            target_lang=other_lang,
            use_find_replace=not bypass_rules
        )
        
        for model_name, result in translation_result["all_results"].items():
            csv_entry = {
                'source': source,
                'target': target,
                'source_lang': source_lang,
                'other_lang': other_lang,
                'translator_name': model_name,
                'translated_text': result.get("translated_text", "[TRANSLATION FAILED]"),
                'cosine_similarity_original_translation': result.get("similarity_vs_original",
                                                                     translation_result["best_result"].get("similarity_vs_original")),
                'cosine_similarity_vs_source': result.get("similarity_vs_source"),
                'cosine_similarity_vs_target': result.get("similarity_vs_target"),
            }
            csv_data.append(csv_entry)
            
            print(f"{f'text out ({language_codes[other_lang]}), predicted with {model_name}:':<{INDENT}}"
                  f"{result.get('translated_text', '[FAILED]')}")
        
        best_result = translation_result["best_result"]
        best_csv_entry = {
            'source': source,
            'target': target,
            'source_lang': source_lang,
            'other_lang': other_lang,
            'translator_name': 'best_model',
            'translated_text': best_result.get("translated_text", "[NO VALID TRANSLATIONS]"),
            'cosine_similarity_original_translation': best_result.get("similarity_vs_original"),
            'cosine_similarity_vs_source': best_result.get("similarity_vs_source"),
            'cosine_similarity_vs_target': best_result.get("similarity_vs_target"),
        }
        csv_data.append(best_csv_entry)
        
        best_model_source = best_result.get("best_model_source", "none")
        print(f"{f'-> best results from {best_model_source}:':<{INDENT}}"
              f"{best_result.get('translated_text', '[NO VALID TRANSLATIONS]')}")
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'source', 'target', 'source_lang', 'other_lang', 'translator_name', 'translated_text',
            'cosine_similarity_original_translation', 'cosine_similarity_vs_source', 'cosine_similarity_vs_target'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)
    
    error_summary = translation_manager.get_error_summary()
    if error_summary["translation_errors"] > 0 or error_summary["find_replace_errors"] > 0:
        with open(errors_path, "w", encoding="utf-8") as f:
            json.dump(error_summary, f, ensure_ascii=False, indent=2)
    
    print(f"\nCompleted test: {name_suffix if name_suffix else 'default'}")
    print(f"Results saved to: {csv_path}")
    print(f"Total samples processed: {n_samples}")
    print(f"Total CSV entries written: {len(csv_data)}")
    
    if error_summary["translation_errors"] > 0:
        print(f"Translation errors: {error_summary['translation_errors']}")
    if error_summary["find_replace_errors"] > 0:
        print(f"Find-replace errors: {error_summary['find_replace_errors']}")
    if error_summary["translation_errors"] > 0 or error_summary["find_replace_errors"] > 0:
        print(f"Error details saved to: {errors_path}")


if __name__ == "__main__":
    random.seed(42)
    os.makedirs("translation_results", exist_ok=True)
    
    training_data = "../Data/training_data.jsonl"
    testing_data = "../Data/testing_data.jsonl"
    merged_model_folder = "../Data/merged/"
    merged_25k_model_folder = "../Data/merged_25k/"
    merged_100k_model_folder = "../Data/merged_100k/"
    
    models_config = {
        "nllb_3b_base_researchonly": {
            "cls": NLLBTranslationModel,
            "params": {
                "base_model_id": "facebook/nllb-200-3.3B",
                "model_type": "seq2seq",
            }
        },
        
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
        "opus_mt_finetuned_25k": {
            "cls": OpusTranslationModel,
            "params": {
                "base_model_id": "Helsinki-NLP/opus-mt-tc-big-en-fr",
                "model_type": "seq2seq",
                "merged_model_path_en_fr": f"{merged_25k_model_folder}opus_mt_en_fr",
                "merged_model_path_fr_en": f"{merged_25k_model_folder}opus_mt_fr_en",
            }
        },
        "opus_mt_finetuned_100k": {
            "cls": OpusTranslationModel,
            "params": {
                "base_model_id": "Helsinki-NLP/opus-mt-tc-big-en-fr",
                "model_type": "seq2seq",
                "merged_model_path_en_fr": f"{merged_100k_model_folder}opus_mt_en_fr",
                "merged_model_path_fr_en": f"{merged_100k_model_folder}opus_mt_fr_en",
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
        "m2m100_418m_finetuned_25k": {
            "cls": M2M100TranslationModel,
            "params": {
                "base_model_id": "facebook/m2m100_418M",
                "model_type": "seq2seq",
                "merged_model_path": f"{merged_25k_model_folder}m2m100_418m",
            }
        },
        "m2m100_418m_finetuned_100k": {
            "cls": M2M100TranslationModel,
            "params": {
                "base_model_id": "facebook/m2m100_418M",
                "model_type": "seq2seq",
                "merged_model_path": f"{merged_100k_model_folder}m2m100_418m",
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
        "mbart50_mmt_finetuned_25k": {
            "cls": MBART50TranslationModel,
            "params": {
                "base_model_id": "facebook/mbart-large-50-many-to-many-mmt",
                "model_type": "seq2seq",
                "merged_model_path_en_fr": f"{merged_25k_model_folder}mbart50_mmt_fr",
                "merged_model_path_fr_en": f"{merged_25k_model_folder}mbart50_mmt_en",
            }
        },
        "mbart50_mmt_finetuned_100k": {
            "cls": MBART50TranslationModel,
            "params": {
                "base_model_id": "facebook/mbart-large-50-many-to-many-mmt",
                "model_type": "seq2seq",
                "merged_model_path_en_fr": f"{merged_100k_model_folder}mbart50_mmt_fr",
                "merged_model_path_fr_en": f"{merged_100k_model_folder}mbart50_mmt_en",
            }
        },
    }
    
    no_token_models = {k: v for k, v in models_config.items() if "_25k" not in k and "_100k" not in k}
    only_token_models = {k: v for k, v in models_config.items() if "_25k" in k or "_100k" in k}
    
    print("\nLoading embedder...")
    embedder = SentenceTransformer('sentence-transformers/LaBSE')
    print("Embedder loaded successfully!\n")
    
    n_tests = 10_000
    print(f"Sampling {n_tests} examples from datasets...")
    sampled_testing_data = sample_data(testing_data, n_tests, use_eval_split=False)
    sampled_training_data = sample_data(training_data, n_tests, use_eval_split=True)
    print(f"Sampled {len(sampled_testing_data)} test examples and {len(sampled_training_data)} train examples\n")
    
    no_rules_manager = TranslationManager(no_token_models, embedder)
    print("Loading models for no-rules evaluation...")
    no_rules_results = no_rules_manager.load_models()
    print(f"Successfully loaded {sum(1 for r in no_rules_results.values() if r['success'])} models")
    
    rules_manager = TranslationManager(only_token_models, embedder)
    print("Loading models for find-replace rules evaluation...")
    rules_results = rules_manager.load_models()
    print(f"Successfully loaded {sum(1 for r in rules_results.values() if r['success'])} models")
    
    test_translations_with_loaded_models(no_rules_manager, sampled_testing_data, "test_no_rules", True)
    test_translations_with_loaded_models(no_rules_manager, sampled_training_data, "train_no_rules", True)
    test_translations_with_loaded_models(rules_manager, sampled_testing_data, "test_rules", False)
    test_translations_with_loaded_models(rules_manager, sampled_training_data, "train_rules", False)
    
    print("\nAll tests completed!")
