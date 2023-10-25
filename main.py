from google.cloud import translate_v2
import html
from langchain.chains.conversation.memory import ConversationBufferMemory
from config import *
from retriever import retrieverQA
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from pprint import pprint


def correct_unicode(text):
    # Convert HTML entities to their respective characters
    corrected_text = html.unescape(text)
    return corrected_text


def translateQuestion(text, source_language="en", target_language='uz'):
    # Initialize the Translate client
    translate_client = translate_v2.Client()

    # Translate the text to the target language (Russian)
    translation = translate_client.translate(
        text,
        source_language=source_language,
        target_language=target_language
    )

    # Extract and return the translated text
    translated_text = translation['translatedText']
    return translated_text


def conversation(openai, retriver, CUSTOM_QUESTION_PROMPT):

    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True,)
    chain = ConversationalRetrievalChain.from_llm(
        openai,
        retriever=retrieverQA,  # see below for vectorstore definition
        memory=memory,
        condense_question_prompt=CUSTOM_QUESTION_PROMPT)

    while True:
        question = input("\nQanday savolingiz mavjud: ")
        question = translateQuestion(question, target_language='en')
        answer = chain({"question": question})
        answer = correct_unicode(
            translateQuestion(answer['answer'], target_language='uz'))
        pprint(f"AI Assistant: {answer}")
        ans = input(
            "\n Yana savolingiz mavjudmi? (ha/yo'q) yoki (stop) yozing)")
        if ans == 'stop' or ans == 'yo\'q':
            break

    pprint(memory.load_memory_variables({})['chat_history'])


if __name__ == "__main__":
    conversation(openai, retriver, CUSTOM_QUESTION_PROMPT)
