"""This module contains code to detect sensitive entities as well as to replace them"""
import hashlib
import logging
import math
from multiprocessing import cpu_count
from os import path
from pathlib import Path

import spacy
from postprocessing.postprocessor import convert_to_pretty
from configuration.configuration import Configuration
from spacy.tokens import DocBin, Token
from tqdm import tqdm

logging.getLogger("transformers").setLevel(logging.WARNING)

N_CPUS = math.floor(cpu_count() / 2)

logger = logging.getLogger(__name__)


class SensitiveTermsRecognizer:
    """
    This class contains code to recognize sensitive terms within texts and make the available to other components as well as
    code to replace those terms in texts.
    """

    def __init__(self, config: Configuration, use_cache: bool = False):
        self.__recognized_sensitive_entities = set()  # set containing sensitive entity types which have been recognized over all texts
        self.__terms = {}
        self.__hashes = {}
        self.__custom_entities = None

        self.__config = config
        self.__use_cache = use_cache

        logger.info("Initializing sensitive terms recognizer with model %s", self.__config.nlp['model'])
        self.__is_transformer = False
        if 'trf' not in self.__config.nlp['model']:
            self.__recognition_function = self.__recognize_using_standard_model
        else:
            self.__recognition_function = self.__recognize_using_transformer_model
            self.__is_transformer = True

        self.__nlp = spacy.load(self.__config.nlp['model'], exclude=["tagger", "parser", "attribute_ruler"])

        cfg = {
            "overwrite_ents": True,
        }

        patterns_path = Path(__file__).parent / "patterns.jsonl"
        ruler = self.__nlp.add_pipe("entity_ruler", config=cfg)
        ruler = ruler.from_disk(patterns_path)

        if self.__config.entities and "custom" in config.entities.keys():
            for custom_entity in config.entities["custom"]:
                ruler.add_patterns(
                    [{"label": custom_entity, "pattern": [{"lower": {"REGEX": config.entities["custom"][custom_entity].lower()}}]}])

            self.__custom_entities = config.entities["custom"].keys()
            logger.info("Custom entities used are %s", ", ".join(self.__custom_entities))

        self.__docs_cache = Path(config.nlp["cache"])
        self.__docs_cache.mkdir(parents=True, exist_ok=True)

    def get_nlp(self):
        """
        Returns language model used to detect sensitive entities
        Returns
        -------
        Language
            Language model.
        """
        return self.__nlp

    def get_recognized_entities(self):
        """
        Returns a set containing all sensitive entity types which have appeared
        Returns
        -------
        set
            Set containing sensitive entity types.
        """
        return self.__recognized_sensitive_entities

    def recognize(self, attribute_name, texts_to_analyze):
        """
        Recognizes sensitive terms in texts and returns them
        Parameters
        ----------
        attribute_name: str
            Name of attribute.
        texts_to_analyze: dict
            Dictionary containing texts and their contexts.
        Returns
        -------
        dict
            Dictionary with the recognized sensitive terms.
        """
        entities_per_id = {}
        n_texts_to_analyze = 0
        n_ids_to_consider = 0
        model_name = self.__nlp.meta['name']

        for person_id in texts_to_analyze:
            n_texts_to_analyze += len(texts_to_analyze[person_id])
            n_ids_to_consider += 1

        logger.info("Recognizing sensitive terms in %d texts of %d ids", n_texts_to_analyze, n_ids_to_consider)
        if self.__use_cache:
            logger.info("Using cached docs if exist")
        else:
            if not self.__is_transformer:
                logger.info("Using %d cpu cores to analyze texts", N_CPUS)

        for person_id in texts_to_analyze:
            # calculate hash using the person_id, the model, and the texts_to_analyze
            calculated_hash = get_hash_of_texts_to_analyze(person_id, model_name, texts_to_analyze[person_id])
            self.__hashes.setdefault(attribute_name, {})[person_id] = calculated_hash

            n_texts_for_id = len(texts_to_analyze[person_id])

            if self.__use_cache and path.exists(path.join(self.__docs_cache, calculated_hash)):
                doc_bin = DocBin().from_disk(path.join(self.__docs_cache, calculated_hash))
                logger.debug("Found %d already processed docs for id %s", len(doc_bin), person_id)
                entities_for_person = self.__get_entities_from_doc_bin(doc_bin)
                entities_per_id.update(entities_for_person)
            else:
                logger.debug("%d ids and %d docs remaining", n_ids_to_consider, n_texts_to_analyze)
                doc_bin = DocBin(attrs=["ENT_IOB", "ENT_TYPE", "ENT_KB_ID", "LEMMA"], store_user_data=True)
                doc_bin, entities_for_person = self.__recognition_function(person_id, texts_to_analyze[person_id], doc_bin)
                entities_per_id.update(entities_for_person)
                doc_bin.to_disk(path.join(self.__docs_cache, calculated_hash))

            n_texts_to_analyze -= n_texts_for_id
            n_ids_to_consider -= 1

            if len(doc_bin) != n_texts_for_id:
                logger.warning("%d texts could not be processed for id %s", n_texts_for_id - len(doc_bin), person_id)

        return entities_per_id

    def replace(self, attribute_name, person_id, replacements, entities_to_remain):
        """
        Replaces entities with their types, replacements from the relational part, or remains them
        Some code taken from https://stackoverflow.com/questions/58712418/replace-entity-with-its-label-in-spacy
        Parameters
        ----------
        attribute_name: str
            Name of attribute.
        person_id: str
            Direct identifier.
        replacements: dict
            Dictionary containing replacements for specific tokens.
        entities_to_remain: set
            Set containing entities which should be kept.
        Returns
        -------
        list
            List with recoded texts.
        """
        calculated_hash = self.__hashes[attribute_name][person_id]
        if not path.exists(path.join(self.__docs_cache, calculated_hash)):
            raise Exception("Could not find processed texts for id {}".format(person_id))

        doc_bin = DocBin().from_disk(path.join(self.__docs_cache, calculated_hash))
        docs = list(doc_bin.get_docs(self.__nlp.vocab))

        replaced_texts = []
        entities_to_consider = self.__config.get_entities_to_consider()
        for doc in docs:
            new_text = doc.text
            for recognized_entity in reversed(doc.ents):
                start = recognized_entity.start_char
                end = start + len(recognized_entity.text)

                # if entity is not relevant, just place original text for this entity
                if recognized_entity.label_ not in entities_to_consider:
                    new_text = new_text[:start] + recognized_entity.text + new_text[end:]
                    continue

                # if entity appears in the entities to remain, also just place original text for this entity
                if recognized_entity.text in [entity.text for entity in entities_to_remain]:
                    new_text = new_text[:start] + recognized_entity.text + new_text[end:]
                    continue

                # if entity type appears in the replacements
                if recognized_entity.label_ in replacements:
                    for entity, to_be_replaced, replacement in replacements[recognized_entity.label_]:
                        # If other doc, just skip it
                        if entity.doc.text != doc.text:
                            continue
                        if recognized_entity.text == entity.text:
                            if isinstance(to_be_replaced, Token):  # if it is a token, only replace this token
                                for term in reversed(recognized_entity):
                                    term_start = term.idx
                                    term_end = term_start + len(term.text)
                                    if term.text == to_be_replaced.text:
                                        new_text = new_text[:term_start] + convert_to_pretty(replacement, self.__config.get_default_date_format()) + new_text[term_end:]
                                    else:
                                        new_text = new_text[:term_start] + term.text + new_text[term_end:]
                            else:  # Not a token, so replace it completely
                                new_text = new_text[:start] + convert_to_pretty(replacement, self.__config.get_default_date_format()) + new_text[end:]
                        else:
                            new_text = new_text[:start] + str(recognized_entity.label_) + new_text[end:]
                    continue

                # if nothing applies, replace entity with its label
                new_text = new_text[:start] + str(recognized_entity.label_) + new_text[end:]

            replaced_texts.append(new_text)
        if len(replaced_texts) == 1:
            return replaced_texts[0]
        return list(replaced_texts)

    def __recognize_using_transformer_model(self, person_id, texts, doc_bin):
        entities = {}
        for (text, index) in tqdm(texts, desc=str(person_id)):
            try:
                entities.setdefault(index, {})
                doc = self.__nlp(text)
                doc.user_data = {}  # Quick fix since TransformerData is not serializable
                doc.user_data["index"] = index
                doc_bin.add(doc)
                entities[index] = self.__get_entities_from_doc(doc)
            except Exception:
                logger.error("Error processing textual attribute at index %s", index, exc_info=True)
        return doc_bin, entities

    def __recognize_using_standard_model(self, person_id, texts, doc_bin):
        entities = {}
        for (doc, index) in tqdm(self.__nlp.pipe(texts, as_tuples=True, batch_size=20, n_process=N_CPUS), total=len(texts), desc=str(person_id)):
            doc.user_data["index"] = index
            entities.setdefault(index, {})
            doc_bin.add(doc)
            entities[index] = self.__get_entities_from_doc(doc)
        return doc_bin

    def __get_entities_from_doc(self, doc):
        ents = {}
        entities_to_consider = self.__config.get_entities_to_consider()
        for entity in doc.ents:
            if entity.label_ in entities_to_consider:
                ents.setdefault(entity.label_, []).append(entity)
                self.__recognized_sensitive_entities.add(entity.label_)
        return ents

    def __get_entities_from_doc_bin(self, doc_bin):
        entities = {}
        docs = list(doc_bin.get_docs(self.__nlp.vocab))
        for doc in docs:
            index = doc.user_data["index"]
            entities.setdefault(index, {})
            entities[index] = self.__get_entities_from_doc(doc)
        return entities


def get_hash_of_texts_to_analyze(person_id, model, texts):
    """
    Takes arguments and builds a fingerprint by calculating a hash
    Parameters
    ----------
    person_id: str
        Direct identifier.
    model: str
        Language model name.
    texts: list
        List of texts.
    Returns
    -------
    str
        Hash.
    """
    text_hash = hashlib.sha256()
    text_hash.update(str(person_id).encode())
    text_hash.update(model.encode())
    for text, _ in texts:
        text_hash.update(text.encode())
    text_hash.update(str(len(texts)).encode())
    return text_hash.hexdigest()
