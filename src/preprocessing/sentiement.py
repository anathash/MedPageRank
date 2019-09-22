import nltk
from nltk.tokenize import sent_tokenize

def main():
    text = """Hello Mr. Smith, how are you doing today? The weather is great, and city is awesome.
    The sky is pinkish-blue. You shouldn't eat cardboard"""
    tokenized_text = sent_tokenize(text)
    print(tokenized_text)

if __name__ == '__main__':
    main()
