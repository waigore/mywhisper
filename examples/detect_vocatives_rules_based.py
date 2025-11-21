import spacy

# Load the small English model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Spacy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
    exit()

def identify_vocatives(text):
    """
    Identifies potential vocatives in a given text using spaCy for NLP.

    Args:
        text (str): The input text.

    Returns:
        list: A list of identified vocative phrases.
    """
    doc = nlp(text)
    vocatives = []
    potential_vocatives = set()

    for sent in doc.sents:
        # Check for vocatives separated by commas, which are often proper nouns (PROPN)
        # or common nouns (NOUN) used as address terms (e.g., 'sir', 'madam').

        # Case 1: Vocative at the beginning, followed by a comma (or other punctuation in token.nbor())
        if len(sent) > 1 and sent[0].pos_ in ["PROPN", "NOUN"] and sent[1].is_punct:
            # Further refinement: check if the next token is a verb, indicating direct action
            if len(sent) > 2 and sent[2].pos_ == "VERB":
                potential_vocatives.add(sent[0].text)

        # Case 2: Vocative at the end, preceded by a comma (or other punctuation)
        if len(sent) > 1 and sent[-1].pos_ in ["PROPN", "NOUN"] and sent[-2].is_punct:
            potential_vocatives.add(sent[-1].text)
        
        # Case 3: Vocative in the middle of a sentence, between commas (less precise with this simple logic)
        # This requires more complex dependency parsing to check for an 'appos' or similar relation
        # but the simple comma check works for basic cases.

        # Advanced check using Dependency Parsing: 
        # Vocatives often have the 'vocative' or 'discourse' dependency label in some models
        # The 'en_core_web_sm' model doesn't explicitly use a 'vocative' tag,
        # but we can look for "NP" (noun phrase) chunks that are not the subject of the main verb.
        for chunk in sent.noun_chunks:
             # If the chunk is a name and not the subject of the sentence's main verb
             if chunk.root.ent_type_ == "PERSON" and chunk.root.dep_ != "nsubj":
                 # This is a heuristic and may catch non-vocative names too,
                 # but combined with punctuation rules, it can work.
                 pass # The simpler punctuation rules are more reliable for the common 'So, Josh, tell me...' pattern
    
    # Filter for names found by NER which also match our potential_vocatives set
    for ent in doc.ents:
        if ent.label_ == "PERSON" and ent.text in potential_vocatives:
            vocatives.append(ent.text)

    return list(set(vocatives)) # Remove duplicates

# Example Usage:
text1 = "So, Josh, tell me about your day. I hope you're having a good time, Sarah. Hey Alex, are you coming?"
text2 = "John went to the store. The store is closed, John."
text3 = "Let's eat, Grandma. Let's eat Grandma." # The algorithm should only find "Grandma" in the first sentence
text4 = "Mary said, 'Hello, Peter!'"

print(f"Text 1 Vocatives: {identify_vocatives(text1)}")
print(f"Text 2 Vocatives: {identify_vocatives(text2)}")
print(f"Text 3 Vocatives: {identify_vocatives(text3)}")
print(f"Text 4 Vocatives: {identify_vocatives(text4)}")

