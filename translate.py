import logging
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel


class BaseTranslationModel:
    def __init__(self, base_model_id, model_type="seq2seq", **parameters):
        self.base_model_id = base_model_id
        self.model_type = model_type
        self.parameters = parameters
        self.model = None
        self.tokenizer = None
        self.finetuned_model = None
        self.special_tokens_added = False
        if self.parameters.get("debug"):
            logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)

    def _get_keep_tokens(self):
        """Generate list of KEEP tokens from <KEEP1> to <KEEP1024>"""
        return [f"<KEEP{i}>" for i in range(1, 1025)]

    def _add_special_tokens(self, tokenizer, model):
        """Add KEEP tokens as special tokens to tokenizer and resize model embeddings"""
        if self.special_tokens_added:
            return

        keep_tokens = self._get_keep_tokens()

        special_tokens_dict = {'additional_special_tokens': keep_tokens}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

        if num_added_toks > 0:
            model.resize_token_embeddings(len(tokenizer))

            keep_token_ids = tokenizer.convert_tokens_to_ids(keep_tokens)

            # Initialize new embeddings to be similar to pad token or a neutral value
            # This helps prevent the model from trying to translate them
            with torch.no_grad():
                if hasattr(model, 'get_input_embeddings'):
                    input_embeddings = model.get_input_embeddings()
                    if tokenizer.pad_token_id is not None:
                        pad_embedding = input_embeddings.weight[tokenizer.pad_token_id].clone()
                        for token_id in keep_token_ids:
                            if token_id is not None:
                                input_embeddings.weight[token_id] = pad_embedding

                if hasattr(model, 'get_output_embeddings'):
                    output_embeddings = model.get_output_embeddings()
                    if tokenizer.pad_token_id is not None and output_embeddings is not None:
                        pad_embedding = output_embeddings.weight[tokenizer.pad_token_id].clone()
                        for token_id in keep_token_ids:
                            if token_id is not None:
                                output_embeddings.weight[token_id] = pad_embedding

        self.special_tokens_added = True
        self.logger.debug(f"Added {num_added_toks} special KEEP tokens to tokenizer")

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

            self._add_special_tokens(tokenizer, self.model)

            # Additional resize check (in case base model vocab was already different)
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
        self.special_tokens_added = False
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
            use_finetuned=False,
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
        self.directional_tokens_added = {}

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

        if cache_key not in self.directional_tokens_added:
            self._add_special_tokens(tokenizer, model)
            self.directional_tokens_added[cache_key] = True

        if hasattr(model.config, "vocab_size") and len(tokenizer) > model.config.vocab_size:
            model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

        self.directional_cache[cache_key] = (tokenizer, model)
        return tokenizer, model

    def translate_text(
            self,
            input_text,
            input_language="en",
            target_language="fr",
            use_finetuned=False,
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

    def clear_cache(self):
        self.directional_cache.clear()
        self.directional_tokens_added.clear()
        super().clear_cache()


class M2M100TranslationModel(BaseTranslationModel):
    LANGUAGE_CODES = {"en": "en", "fr": "fr"}

    def translate_text(self, input_text, input_language="en", target_language="fr",
                       use_finetuned=False, generation_kwargs=None):
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


class MBART50TranslationModel(BaseTranslationModel):
    LANGUAGE_CODES = {"en": "en_XX", "fr": "fr_XX"}

    def __init__(self, base_model_id, model_type="seq2seq", **parameters):
        super().__init__(base_model_id, model_type, **parameters)
        self.directional_cache = {}
        self.directional_tokens_added = {}

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

        if cache_key not in self.directional_tokens_added:
            self._add_special_tokens(tokenizer, model)
            self.directional_tokens_added[cache_key] = True

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
        self.directional_tokens_added.clear()
        super().clear_cache()


def create_translator(translator_class, **config):
    return translator_class(**config)
