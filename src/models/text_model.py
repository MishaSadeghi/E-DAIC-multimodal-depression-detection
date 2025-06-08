from transformers import AutoTokenizer, AutoModelForSequenceClassification

def build_deproberta_model(model_name="rafalposwiata/deproberta-large-depression", num_labels=3):
    """
    Loads the DepRoBERTa model and tokenizer from Hugging Face.

    Args:
        model_name (str): The name of the model on the Hugging Face Hub.
        num_labels (int): The number of output labels for the classifier.

    Returns:
        A tuple containing the model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    
    return model, tokenizer 