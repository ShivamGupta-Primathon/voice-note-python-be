import logging
import os
import nltk
import pandas as pd
import re
from dotenv import load_dotenv, find_dotenv
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from openai import OpenAI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from langdetect import detect

_ = load_dotenv(find_dotenv())

OPEN_API_KEY = os.getenv('OPEN_API_KEY')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Transcript(BaseModel):
    transcript: str


app = FastAPI()

# Initialize OpenAI client
client = OpenAI(api_key=OPEN_API_KEY)

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def lemmatize_text(text: str) -> str:
    try:
        lemmatizer = WordNetLemmatizer()
        words = nltk.word_tokenize(text)
        lemmatized = [lemmatizer.lemmatize(word) for word in words if word.isalnum()]
        return ' '.join(lemmatized)
    except Exception as e:
        logger.error(f"Error lemmatizing text: {e}")
        raise


def vectorize_text(text: str) -> pd.DataFrame:
    try:
        vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
        vectors = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        dense = vectors.todense()
        denselist = dense.tolist()
        df = pd.DataFrame(denselist, columns=feature_names)
        return df
    except Exception as e:
        logger.error(f"Error vectorizing text: {e}")
        raise


def ask_question(messages: list) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error occurred")


def detect_language(text: str) -> str:
    try:
        return detect(text)
    except Exception as e:
        logger.error(f"Error detecting language: {e}")
        raise HTTPException(status_code=500, detail="Error detecting language")


def detect_task_type(text: str) -> str:
    try:
        text_lower = text.lower()

        # Define patterns for different task types
        list_keywords = ["list", "tasks", "items"]
        journal_keywords = ["journal", "entry", "diary"]
        email_keywords = ["email", "mail", "letter", "correspondence"]
        meeting_keywords = ["meeting", "conference", "discussion", "call", "session", "briefing"]
        sales_call_keywords = ["sales call", "sales meeting", "client call", "sales discussion", "business call"]

        # Match against patterns
        if any(re.search(r'\b' + keyword + r'\b', text_lower) for keyword in list_keywords):
            return "list"
        elif any(re.search(r'\b' + keyword + r'\b', text_lower) for keyword in journal_keywords):
            return "journal"
        elif any(re.search(r'\b' + keyword + r'\b', text_lower) for keyword in email_keywords):
            return "email"
        elif any(re.search(r'\b' + keyword + r'\b', text_lower) for keyword in meeting_keywords):
            # Check for context to avoid casual mentions
            if re.search(r'\b(agenda|minutes|discussed|summary|report|attendees|actions|agreed)\b', text_lower):
                return "meeting"
        elif any(re.search(r'\b' + keyword + r'\b', text_lower) for keyword in sales_call_keywords):
            # Check for context to avoid casual mentions
            if re.search(r'\b(agenda|minutes|discussed|summary|report|client|actions|agreed)\b', text_lower):
                return "sales_call"
        else:
            return "other"  # Default to "other" if type cannot be determined
    except Exception as e:
        logger.error(f"Error detecting task type: {e}")
        raise HTTPException(status_code=500, detail="Error detecting task type")


def generate_title(transcript: str) -> str:
    try:
        # Vectorize the transcript
        vectorized_text = vectorize_text(transcript)

        # Get the most important words from the vectorized text
        important_words = ' '.join(vectorized_text.columns[:10])  # Adjust the number of words as needed

        # Create a detailed prompt using the important words
        prompt = (
            f"Generate response in english"
            f"Generate a short and crisp title for the following text. The title should be brief and contextually related to the task described in the text. "
            f"If the text is a journal entry, the title should reflect the date or main event. "
            f"If the text is a list, the title should reflect the type of list. "
            f"If the text is something else, the title should capture the main theme or purpose. "
            f"Please avoid including specific values or list items in the title or including details. "
            f"Here are the most important words to consider: {important_words}. "
            f"Here is the text: {transcript}."
        )

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        # Call OpenAI API to generate title
        response = ask_question(messages)

        # Extract and return the generated title
        generated_title = response.strip()  # Assuming GPT-4 returns a single line response
        return generated_title
    except Exception as e:
        logger.error(f"Error generating title: {e}")
        raise HTTPException(status_code=500, detail="Error generating title")


def create_analysis(transcription):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",
         "content": f"Please analyze the following transcription and provide key insights. Here is the transcription: {transcription}"}
    ]
    return ask_question(messages)


def generate_defined_data(transcript: str, target_language: str) -> str:
    try:
        # Determine the type of content
        task_type = detect_task_type(transcript)

        # Generate prompt based on detected task type
        if task_type == "list":
            prompt = (
                f"Generate a response for the following text into {target_language}. "
                f"remove any noise from the transcript and only include taks or items as bullet points using HTML tags <ul> and <li>. "
                f"perform the task asked in the transcript on the items"
                f"Remove any unnecessary words or phrases such as 'add', 'buy', 'get', etc. "
                f"Here is the text: {transcript}"
            )
        elif task_type == "journal":
            prompt = (
                f"Generate a response for the following text into {target_language}. "
                f"Please summarize it as a journal entry, focusing only on the main points or themes. "
                f"Use paragraph tags <p> for formatting. Here is the text: {transcript}"
            )
        elif task_type == "email":
            prompt = (
                f"Generate response for the following text into {target_language}. Compose an email based on the text. Include necessary details and format it appropriately for sending. "
                f"Use proper paragraph tags <p> and line breaks <br> as needed. Here is the text: {transcript}"
            )
        elif task_type in ["meeting", "sales_call"]:
            # Prompt for generating a meeting report
            prompt = (
                f"Generate a detailed meeting report in {target_language} based on the transcript provided. Include the following sections formatted with HTML tags:\n\n"
                f"<h2>Heading or Title:</h2>\n"
                f"<p>- Date and time of the meeting</p>\n"
                f"<p>- Subject or purpose of the meeting</p>\n\n"
                f"<h3>Introduction:</h3>\n"
                f"<p>- Brief overview of the meeting's purpose and attendees</p>\n\n"
                f"<h3>Bullet Points or Agenda:</h3>\n"
                f"<ul>\n"
                f"  <li>List of key agenda items discussed during the meeting</li>\n"
                f"</ul>\n\n"
                f"<h3>Summary of Discussion:</h3>\n"
                f"<p>- Detailed account of discussions under each agenda item</p>\n"
                f"<ul>\n"
                f"  <li>Include important points raised, decisions made, and actions agreed upon</li>\n"
                f"  <li>Use bullet points or numbered lists for clarity</li>\n"
                f"</ul>\n\n"
                f"<h3>Analysis:</h3>\n"
                f"<ul>\n"
                f"  <li>Analyze the implications of decisions made or topics discussed</li>\n"
                f"  <li>Evaluate any challenges, opportunities, or risks identified</li>\n"
                f"  <li>Provide insights or interpretations where relevant</li>\n"
                f"</ul>\n\n"
                f"<h3>Conclusion:</h3>\n"
                f"<ul>\n"
                f"  <li>Summarize the overall outcome of the meeting</li>\n"
                f"  <li>Highlight key takeaways and action points</li>\n"
                f"  <li>Clearly state any follow-up actions or next steps</li>\n"
                f"</ul>\n\n"
                f"<h3>Recommendations (if applicable):</h3>\n"
                f"<ul>\n"
                f"  <li>Offer suggestions for future actions or improvements based on the meeting's outcomes</li>\n"
                f"</ul>\n\n"
                f"<h3>Appendix (if needed):</h3>\n"
                f"<p>- Attach additional documents, charts, or data referenced during the meeting</p>\n\n"
                f"Tips for Formatting:\n"
                f"- Use headings and subheadings to organize different sections clearly.\n"
                f"- Use html tags bullet points or numbered lists for agenda items, summaries, and action points to enhance readability.\n"
                f"- Keep paragraphs concise and focused to ensure clarity and maintain attention.\n\n"
                f"Here is the text: {transcript}"
            )
        else:
            prompt = (
                f"Generate a response for the following text into {target_language}. "
                f"remove all noise and unnecessary sentences and only extract main points"
                f"Please list the tasks or items as bullet points using HTML tags <ul> and <li>. "
                f"Here is the text: {transcript}"
            )

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        response = ask_question(messages)

        # Assuming GPT-4 returns a well-formatted response
        return response.strip()
    except Exception as e:
        logger.error(f"Error generating defined data: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while generating defined data")


@app.get('/')
def get_notes() -> Dict[str, str]:
    return {"message": "Welcome to the transcription and analysis API!"}


@app.post('/process-text', response_model=Dict[str, Any])
async def process_text(request: Transcript) -> Dict[str, Any]:
    try:
        transcript = request.transcript
        target_language = "en"  # Specify the target language here, e.g., "en" for English

        lemmatized_text = lemmatize_text(transcript)
        tfidf_vector = vectorize_text(lemmatized_text)
        title = generate_title(lemmatized_text)
        html_formatted = generate_defined_data(lemmatized_text, target_language)

        tfidf_vector_str = tfidf_vector.to_json()

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",
             "content": f"Please provide an appropriate response based on the following transcription: {transcript}. Here is some vectorized text that provides additional context: {tfidf_vector_str}"}
        ]
        analytical_response = create_analysis(messages)

        return {
            'title': title,
            'html_formatted': html_formatted,
            'analytical_response': analytical_response
        }
    except HTTPException as http_exc:
        logger.error(f"HTTP error: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logger.error(f"Error processing text: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the text")
