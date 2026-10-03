"""One builder module per training source. DATASETS maps each module name to its entry in data/licenses.json."""

DATASETS = {"vitaminc": "tals/vitaminc", "snli": "stanfordnlp/snli", "wanli": "alisawuffles/WANLI",
            "ai2_arc": "allenai/ai2_arc", "banking77": "PolyAI/banking77", "clinc_oos": "clinc/clinc_oos",
            "massive": "AmazonScience/massive", "typed_decisions": "LocalLLaMA/typed-decisions",
            "civil_comments": "google/civil_comments", "boolq": "google/boolq", "helpsteer2": "nvidia/HelpSteer2"}
