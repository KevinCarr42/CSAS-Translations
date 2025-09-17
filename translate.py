import logging
import torch
from sentence_transformers.util import pytorch_cos_sim
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, BitsAndBytesConfig

from text_processing import preprocess_for_translation, postprocess_translation


class BaseTranslationModel:
    def __init__(self, base_model_id, model_type="seq2seq", **parameters):
        self.base_model_id = base_model_id
        self.model_type = model_type
        self.parameters = parameters
        self.model = None
        self.tokenizer = None
        self.finetuned_model = None
        if self.parameters.get("debug"):
            logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
    
    def _tokenizer_kwargs(self):
        return {
            "use_fast": True,
            "local_files_only": self.parameters.get("local_files_only", False),
        }
    
    def _model_kwargs(self, allow_device_map=True):
        kwargs = {
            "trust_remote_code": True,
            "local_files_only": self.parameters.get("local_files_only", False),
            "torch_dtype": self.parameters.get("dtype", torch.bfloat16),
        }
        if allow_device_map:
            kwargs["device_map"] = self.parameters.get("device_map", "auto")
            kwargs["offload_folder"] = self.parameters.get("offload_folder", "./offload")
            if self.parameters.get("max_memory"):
                kwargs["max_memory"] = self.parameters["max_memory"]
        if self.parameters.get("revision"):
            kwargs["revision"] = self.parameters["revision"]
        if self.parameters.get("use_quantization"):
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.parameters.get("dtype", torch.bfloat16),
            )
        return kwargs
    
    def load_tokenizer(self):
        if self.tokenizer is None:
            tokenizer_path = self.parameters.get("merged_model_path", self.base_model_id)
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, **self._tokenizer_kwargs()
            )
            if getattr(self.tokenizer, "pad_token", None) is None and getattr(
                    self.tokenizer, "eos_token", None
            ):
                self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer
    
    def load_model(self):
        if self.model is None:
            loader = AutoModelForSeq2SeqLM if self.model_type == "seq2seq" else AutoModelForCausalLM
            
            model_path = self.parameters.get("merged_model_path", self.base_model_id)
            
            self.model = loader.from_pretrained(
                model_path, **self._model_kwargs(allow_device_map=True)
            )
            tokenizer = self.load_tokenizer()
            if hasattr(self.model.config, "vocab_size") and len(tokenizer) > self.model.config.vocab_size:
                self.model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        return self.model
    
    def translate_text(self, input_text, input_language="en", target_language="fr",
                       generation_kwargs=None):
        tokenizer = self.load_tokenizer()
        model = self.load_model()
    
    def clean_output(self, text):
        import re
        patterns = [
            r"^(Here is the translation|Voici la traduction)[:\s]*",
            r"^(Translation|Traduction)[:\s]*",
            r"^(The translation is|La traduction est)[:\s]*",
            r"\s*\([^)]*translation[^)]*\)\s*$",
        ]
        cleaned = text
        for pattern in patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
        return cleaned
    
    def clear_cache(self):
        self.model = None
        self.finetuned_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class NLLBTranslationModel(BaseTranslationModel):
    # NOTE: research only license for this model
    
    LANGUAGE_CODES = {"en": "eng_Latn", "fr": "fra_Latn"}
    
    def translate_text(
            self,
            input_text,
            input_language="en",
            target_language="fr",
            generation_kwargs=None,
    ):
        tokenizer = self.load_tokenizer()
        model = self.load_model()
        
        source_code = self.LANGUAGE_CODES[input_language]
        target_code = self.LANGUAGE_CODES[target_language]
        tokenizer.src_lang = source_code
        
        model_inputs = tokenizer(input_text, return_tensors="pt", padding=True)
        model_inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in model_inputs.items()}
        
        target_token_id = tokenizer.convert_tokens_to_ids(target_code)
        
        generation_arguments = {
            "max_new_tokens": 256,
            "num_beams": 4,
            "do_sample": False,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if target_token_id is not None:
            generation_arguments["forced_bos_token_id"] = target_token_id
        if generation_kwargs:
            generation_arguments.update(generation_kwargs)
        
        output_token_ids = model.generate(**model_inputs, **generation_arguments)
        text_output = tokenizer.batch_decode(output_token_ids, skip_special_tokens=True)[0].strip()
        return self.clean_output(text_output)


class OpusTranslationModel(BaseTranslationModel):
    LANGUAGE_ALIASES = {"en": "en", "fr": "fr"}
    
    def __init__(self, base_model_id, model_type="seq2seq", **parameters):
        super().__init__(base_model_id, model_type, **parameters)
        self.directional_cache = {}
    
    def _root_model_id(self):
        parts = self.base_model_id.split("-")
        if parts[-2:] in (["en", "fr"], ["fr", "en"]):
            return "-".join(parts[:-2])
        return self.base_model_id
    
    def _directional_model_id(self, source_language, target_language):
        root_id = self._root_model_id()
        source_alias = self.LANGUAGE_ALIASES[source_language]
        target_alias = self.LANGUAGE_ALIASES[target_language]
        return f"{root_id}-{source_alias}-{target_alias}"
    
    def _load_directional(self, source_language, target_language):
        cache_key = f"{source_language}-{target_language}"
        if cache_key in self.directional_cache:
            return self.directional_cache[cache_key]
        
        merged_path = self.parameters.get(f"merged_model_path_{source_language}_{target_language}")
        model_id = merged_path if merged_path else self._directional_model_id(source_language, target_language)
        
        tokenizer = AutoTokenizer.from_pretrained(model_id, **self._tokenizer_kwargs())
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id, **self._model_kwargs(allow_device_map=False)
        )
        if torch.cuda.is_available():
            model = model.cuda()
        if hasattr(model.config, "vocab_size") and len(tokenizer) > model.config.vocab_size:
            model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        
        self.directional_cache[cache_key] = (tokenizer, model)
        return tokenizer, model
    
    def translate_text(
            self,
            input_text,
            input_language="en",
            target_language="fr",
            generation_kwargs=None,
    ):
        tokenizer, model = self._load_directional(input_language, target_language)
        
        model_inputs = tokenizer(input_text, return_tensors="pt", padding=True)
        model_inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in model_inputs.items()}
        
        generation_arguments = {
            "max_new_tokens": 256,
            "num_beams": 4,
            "do_sample": False,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if generation_kwargs:
            generation_arguments.update(generation_kwargs)
        
        output_token_ids = model.generate(**model_inputs, **generation_arguments)
        text_output = tokenizer.batch_decode(output_token_ids, skip_special_tokens=True)[0].strip()
        return self.clean_output(text_output)


class M2M100TranslationModel(BaseTranslationModel):
    LANGUAGE_CODES = {"en": "en", "fr": "fr"}
    
    def translate_text(self, input_text, input_language="en", target_language="fr", generation_kwargs=None):
        tokenizer = self.load_tokenizer()
        model = self.load_model()

        source_code = self.LANGUAGE_CODES[input_language]
        target_code = self.LANGUAGE_CODES[target_language]
        tokenizer.src_lang = source_code

        model_inputs = tokenizer(input_text, return_tensors="pt", padding=True)
        model_inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in model_inputs.items()}

        generation_arguments = {
            "max_new_tokens": 256,
            "num_beams": 4,
            "do_sample": False,
            "pad_token_id": tokenizer.pad_token_id,
            "forced_bos_token_id": tokenizer.get_lang_id(target_code),
        }
        if generation_kwargs:
            generation_arguments.update(generation_kwargs)

        output_token_ids = model.generate(**model_inputs, **generation_arguments)
        text_output = tokenizer.batch_decode(output_token_ids, skip_special_tokens=True)[0].strip()
        return self.clean_output(text_output)
    
    # TODO: DELETE AFTER TESTING
    # def translate_text(self, input_text, input_language="en", target_language="fr", generation_kwargs=None):
    #     tokenizer = self.load_tokenizer()
    #     model = self.load_model()
    #
    #     source_code = self.LANGUAGE_CODES[input_language]
    #     target_code = self.LANGUAGE_CODES[target_language]
    #     tokenizer.src_lang = source_code
    #
    #     model_inputs = tokenizer(input_text, return_tensors="pt", padding=True)
    #     model_inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in model_inputs.items()}
    #
    #     generation_arguments = {
    #         "max_new_tokens": 256,
    #         "num_beams": 4,
    #         "do_sample": False,
    #         "pad_token_id": tokenizer.pad_token_id,
    #         "forced_bos_token_id": tokenizer.get_lang_id(target_code),
    #     }
    #
    #     print(f"BEFORE update - generation_kwargs type: {type(generation_kwargs)}, value: {generation_kwargs}")
    #
    #     if generation_kwargs:
    #         print(f"BEFORE update: {generation_arguments}")
    #         generation_arguments.update(generation_kwargs)
    #         print(f"AFTER update: {generation_arguments}")
    #
    #     # CRITICAL TEST - manually override to prove it works
    #     # generation_arguments["max_new_tokens"] = 5
    #     # print(f"MANUALLY FORCED max_new_tokens=5")
    #
    #     print(f"CALLING generate() with: {generation_arguments}")
    #     output_token_ids = model.generate(**model_inputs, **generation_arguments)
    #
    #     print(f"Output shape: {output_token_ids.shape}")
    #     print(f"Output tokens: {output_token_ids[0][:20]}")  # First 20 tokens
    #
    #     text_output = tokenizer.batch_decode(output_token_ids, skip_special_tokens=True)[0].strip()
    #     print(f"Decoded text length: {len(text_output)}")
    #     print(f"First 50 chars: {text_output[:50]}")
    #
    #     return self.clean_output(text_output)


class MBART50TranslationModel(BaseTranslationModel):
    LANGUAGE_CODES = {"en": "en_XX", "fr": "fr_XX"}
    
    def __init__(self, base_model_id, model_type="seq2seq", **parameters):
        super().__init__(base_model_id, model_type, **parameters)
        self.directional_cache = {}
    
    def _get_directional_model_path(self, source_language, target_language):
        direction_key = f"merged_model_path_{source_language}_{target_language}"
        if direction_key in self.parameters:
            return self.parameters[direction_key]
        
        return self.base_model_id
    
    def _load_directional(self, source_language, target_language):
        cache_key = f"{source_language}-{target_language}"
        if cache_key in self.directional_cache:
            return self.directional_cache[cache_key]
        
        model_path = self._get_directional_model_path(source_language, target_language)
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, **self._tokenizer_kwargs())
        if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None):
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path, **self._model_kwargs(allow_device_map=False)
        )
        if torch.cuda.is_available():
            model = model.cuda()
        
        if hasattr(model.config, "vocab_size") and len(tokenizer) > model.config.vocab_size:
            model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        
        self.directional_cache[cache_key] = (tokenizer, model)
        return tokenizer, model
    
    def translate_text(self, input_text, input_language="en", target_language="fr",
                       generation_kwargs=None):
        tokenizer, model = self._load_directional(input_language, target_language)
        
        source_code = self.LANGUAGE_CODES[input_language]
        target_code = self.LANGUAGE_CODES[target_language]
        tokenizer.src_lang = source_code
        
        model_inputs = tokenizer(input_text, return_tensors="pt", padding=True)
        model_inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in model_inputs.items()}
        
        target_id = getattr(tokenizer, "lang_code_to_id", {}).get(target_code) if hasattr(tokenizer, "lang_code_to_id") else None
        if target_id is None:
            target_id = tokenizer.convert_tokens_to_ids(target_code)
        
        generation_arguments = {
            "max_new_tokens": 256,
            "num_beams": 4,
            "do_sample": False,
            "pad_token_id": tokenizer.pad_token_id,
            "forced_bos_token_id": target_id,
        }
        if generation_kwargs:
            generation_arguments.update(generation_arguments)
        
        output_token_ids = model.generate(**model_inputs, **generation_arguments)
        text_output = tokenizer.batch_decode(output_token_ids, skip_special_tokens=True)[0].strip()
        return self.clean_output(text_output)
    
    def clear_cache(self):
        self.directional_cache.clear()
        super().clear_cache()


class TranslationManager:
    TOKEN_PREFIXES = ['NOMENCLATURE', 'TAXON', 'ACRONYM', 'SITE']
    
    def __init__(self, all_models, embedder=None):
        self.all_models = all_models
        self.embedder = embedder
        self.loaded_models = {}
        self.find_replace_errors = {}
        self.extra_token_errors = {}
    
    def load_models(self, model_names=None):
        if model_names is None:
            model_names = list(self.all_models.keys())
        
        for name in model_names:
            config = self.all_models[name]
            model_instance = config['cls'](**config.get('params', {}))
            _ = model_instance.translate_text("Test", "en", "fr")
            self.loaded_models[name] = model_instance
    
    def translate_with_retries(self, model, text, source_lang, target_lang,
                               token_mapping=None, base_generation_kwargs=None):
        param_variations = [
            {"num_beams": 4},
            {"num_beams": 2},
            {"num_beams": 5},
            {"num_beams": 6},
            {"num_beams": 7},
            {"num_beams": 8},
            {"num_beams": 4, "length_penalty": 0.8},
            {"num_beams": 4, "length_penalty": 1.2},
            {"num_beams": 4, "repetition_penalty": 1.1},
            # TODO: delete after testing
            {"max_new_tokens": 10},
            {"max_new_tokens": 5},
            {"min_length": 200},
        ]
        
        base_kwargs = base_generation_kwargs or {}
        
        for i, params in enumerate(param_variations):
            generation_kwargs = {**base_kwargs, **params}
            
            translated = model.translate_text(
                text, source_lang, target_lang, generation_kwargs=generation_kwargs
            )
            
            if self.is_valid_translation(translated, text, token_mapping):
                print(f'VALID :) ({i})')  # TODO: delete after testing
                return translated, i, params
            else:  # TODO: delete after testing
                print(f'invalid :( ({i}):\t', params)
                print('\t\t\t', translated)
        
        return None, len(param_variations), None
    
    def check_token_prefix_error(self, translated_text, original_text):
        for token_prefix in self.TOKEN_PREFIXES:
            if token_prefix in translated_text:
                if not original_text or token_prefix not in original_text:
                    return True
        return False
    
    def is_valid_translation(self, translated_text, original_text, token_mapping=None):
        if self.check_token_prefix_error(translated_text, original_text):
            return False
        
        if token_mapping:
            for key in token_mapping.keys():
                if key not in translated_text:
                    return False
        
        return True
    
    def translate_single(self, text, model_name, source_lang="en", target_lang="fr",
                         use_find_replace=True, generation_kwargs=None, idx=None,
                         target_text=None, debug=False):
        model = self.loaded_models[model_name]
        
        find_replace_error = False
        retry_attempts = 0
        retry_params = None
        
        if use_find_replace:
            preprocessed_text, token_mapping = preprocess_for_translation(text)
            
            translated_with_tokens, retry_attempts, retry_params = self.translate_with_retries(
                model, preprocessed_text, source_lang, target_lang,
                token_mapping, generation_kwargs
            )
            
            if translated_with_tokens and self.is_valid_translation(
                    translated_with_tokens, text, token_mapping
            ):
                translated_text = postprocess_translation(
                    translated_with_tokens, token_mapping
                )
            else:
                find_replace_error = True
                if debug:
                    self.find_replace_errors[f"{model_name}_{idx}"] = {
                        "original_text": text,
                        "preprocessed_text": preprocessed_text,
                        "translated_with_tokens": translated_with_tokens,
                        "token_mapping": token_mapping,
                        "retry_attempts": retry_attempts,
                        "final_retry_params": retry_params,
                    }
                translated_text = model.translate_text(
                    text, source_lang, target_lang, generation_kwargs
                )
        else:
            preprocessed_text = None
            translated_with_tokens = None
            token_mapping = None
            translated_text = model.translate_text(
                text, source_lang, target_lang, generation_kwargs
            )
        
        token_prefix_error = self.check_token_prefix_error(translated_text, text)
        if token_prefix_error and debug:
            tokens_to_replace = [x for x in token_mapping.keys()] if token_mapping else None
            self.extra_token_errors[f"{model_name}_{idx}"] = {
                "original_text": text,
                "translated_text": translated_text,
                "use_find_replace": use_find_replace,
                "tokens_to_replace": tokens_to_replace,
                "preprocessed_text": preprocessed_text,
                "translated_with_tokens": translated_with_tokens,
                "retry_attempts": retry_attempts,
                "final_retry_params": retry_params,
            }
        
        source_embedding = self.embedder.encode(text, convert_to_tensor=True)
        translated_embedding = self.embedder.encode(translated_text, convert_to_tensor=True)
        similarity_vs_source = pytorch_cos_sim(source_embedding, translated_embedding).item()
        
        similarity_vs_target = None
        similarity_of_original_translation = None
        if target_text:
            target_embedding = self.embedder.encode(target_text, convert_to_tensor=True)
            similarity_vs_target = pytorch_cos_sim(target_embedding, translated_embedding).item()
            similarity_of_original_translation = pytorch_cos_sim(source_embedding, target_embedding).item()
        
        return {
            "find_replace_error": find_replace_error,
            "token_prefix_error": token_prefix_error,
            "translated_text": translated_text,
            "similarity_of_original_translation": similarity_of_original_translation,
            "similarity_vs_source": similarity_vs_source,
            "similarity_vs_target": similarity_vs_target,
            "model_name": model_name,
            "retry_attempts": retry_attempts if use_find_replace else 0,
        }
    
    def translate_with_all_models(self, text, source_lang="en", target_lang="fr",
                                  use_find_replace=True, generation_kwargs=None,
                                  idx=None, target_text=None, debug=False):
        model_names = list(self.loaded_models.keys())
        
        all_results = {}
        best_result = None
        best_similarity = float('-inf')
        
        for model_name in model_names:
            result = self.translate_single(
                text, model_name, source_lang, target_lang,
                use_find_replace, generation_kwargs, idx, target_text, debug
            )
            all_results[model_name] = result
            
            if self.is_valid_translation(result['translated_text'], text) and result["similarity_vs_source"] is not None:
                if result["similarity_vs_source"] > best_similarity:
                    best_similarity = result["similarity_vs_source"]
                    best_result = result.copy()
                    best_result["model_name"] = "best_model"
                    best_result["best_model_source"] = model_name
        
        if best_result is None:
            best_result = {
                "error": "No valid translations from any model",
                "translated_text": "[NO VALID TRANSLATIONS]",
                "similarity_vs_source": None,
                "similarity_vs_target": None,
                "model_name": "best_model",
                "best_model_source": None
            }
        
        all_results['best_model'] = best_result
        
        # TODO: option to just return best model results without the extra info
        return all_results
    
    def get_error_summary(self):
        return {
            "extra_token_errors": len(self.extra_token_errors),
            "find_replace_errors": len(self.find_replace_errors),
            "extra_token_error_details": self.extra_token_errors,
            "find_replace_error_details": self.find_replace_errors,
        }
    
    def clear_errors(self):
        self.extra_token_errors.clear()
        self.find_replace_errors.clear()


def create_translator(translator_class, **config):
    return translator_class(**config)
