import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# # Load the sentiment analysis model that includes a neutral category
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def analyze_sentiment(text):
    result = sentiment_analyzer(text)[0]
    label_mapping = {
        'LABEL_0': 'negative',
        'LABEL_1': 'neutral',
        'LABEL_2': 'positive'
    }
    sentiment = label_mapping[result['label'].upper()]
    if sentiment == 'neutral':
        sentiment = 'null'
    return sentiment

with st.form("my_form"):
    st.title('Feelings2Words')
    text_review = st.text_area('Write me a review') 

    submitted = st.form_submit_button("Submit")
    if submitted:
        template = """
        Please act as a machine learning model trained to perform a supervised learning task,
        to extract the sentiment of a review in '{option}' Language.

        Give your answer writing a JSON evaluating the sentiment field between the dollar sign, the value must be printed without dollar sign.
        The value of sentiment must be "positive" or "negative", otherwise if the text is not valuable write "null".

        Example:

        field 1 named :
        text_review with value: {text_review}
        field 2 named :
        sentiment with value: $sentiment$
        Field 3 named : 
        language with value: {option}
        Review text: '''{text_review}'''

        """

        prompt = PromptTemplate(template=template, input_variables=["text_review", "option"])

        sentiment = analyze_sentiment(text_review)

        response = {
            "text_review": text_review,
            "sentiment": sentiment,
            "language": "English"
        }

        st.json(response)
        print(response)
