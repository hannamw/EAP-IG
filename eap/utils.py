display_name_dict = {
    'ioi': 'IOI', 
    'greater-than': "Greater-Than (GT)", 
    "greater-than-multitoken": "Greater-Than",
    'greater-than-price': "GT (Price)", 
    'greater-than-sequence': "GT (Sequence)", 
    'gender-bias': 'Gender-Bias', 
    'gendered-pronoun': 'Gendered Pronoun',
    'math': 'Math',
    'math-add': 'Math (Addition)',
    'math-sub': 'Math (Subtraction)',
    'math-mul': 'Math (Multiplication)',
    'sva': 'SVA', 
    'fact-retrieval-comma': 'Capital-Country', 
    'fact-retrieval-rev': 'Country-Capital', 
    'hypernymy-comma': 'Hypernymy',
    'hypernymy': 'Hypernymy',
    'npi': 'NPI',
    'colored-objects': 'Colored Objects',
    'entity-tracking': 'Entity Tracking',
    'wug': 'Wug Test',
    'echo': 'Echo',
    'fact-retrieval-rev-multilingual-en': 'Country-Capital (EN)',
    'fact-retrieval-rev-multilingual-nl': 'Country-Capital (NL)',
    'fact-retrieval-rev-multilingual-fr': 'Country-Capital (FR)',
    'fact-retrieval-rev-multilingual-de': 'Country-Capital (DE)',
    'sva-multilingual-en': 'SVA (EN)',
    'sva-multilingual-nl': 'SVA (NL)',
    'sva-multilingual-fr': 'SVA (FR)',
    'sva-multilingual-de': 'SVA (DE)',
    'counterfact-citizen_of': 'FR (Citizen Of)', 
    'counterfact-official_language': 'FR (Official Language)', 
    'counterfact-has_profession': 'FR (Has Profession)', 
    'counterfact-plays_instrument': 'FR (Plays Instrument)', 
    'counterfact-all': 'FR (All)',
}

def model2family(model_name: str):
    if 'gpt2' in model_name:
        return 'gpt2'
    elif 'pythia' in model_name:
        return 'pythia'
    elif 'llama-3' in model_name.lower():
        return 'llama-3'
    elif 'qwen2' in model_name.lower():
        return 'qwen2'
    elif 'gemma' in model_name.lower():
        return 'gemma'
    elif 'mistral' in model_name.lower():
        return 'mistral'
    elif 'olmo' in model_name.lower():
        return 'olmo'
    else:
        raise ValueError(f"Couldn't find model family for model: {model_name}")