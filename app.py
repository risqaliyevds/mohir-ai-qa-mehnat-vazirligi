import gradio as gr
import os
import time
from google.cloud import translate_v2
import html
from langchain.chains.conversation.memory import ConversationBufferMemory
from config import *
from retriver import retriver
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from pprint import pprint
from main import correct_unicode, translateQuestion


memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)

chain = ConversationalRetrievalChain.from_llm(
    openai,
    retriever=retriver,  # see below for vectorstore definition
    memory=memory,
    condense_question_prompt=CUSTOM_QUESTION_PROMPT)


def add_text(history, text):
    history = history + [(text, None)]
    return history, gr.Textbox(value="", interactive=False)


def bot(history, lang_dropdown):
    response = history[-1][0]
    history[-1][1] = ""

    if lang_dropdown == "UZ":
        question_uz = response
        question_en = translateQuestion(
            question_uz, source_language="uz", target_language='en')
        answer_en = chain({"question": question_en})
        answer_uz = correct_unicode(
            translateQuestion(answer_en['answer'], source_language="en", target_language='uz'))
        response = answer_uz
    else:
        answer_en = chain({"question": response})
        response = answer_en['answer']

    for character in response:
        history[-1][1] += character
        time.sleep(0.05)
        yield history


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        avatar_images=(
            None, (os.path.join(os.path.dirname(__file__), "source/logo.jpg"))),
    )

    with gr.Row():
        # Add dropdown for language selection
        lang_dropdown = gr.Dropdown(
            ["ENG", "UZ"], default="UZ", label="Select language")
        txt = gr.Textbox(
            scale=4,
            show_label=False,
            placeholder="Enter text and press enter!",
            container=False,
        )

    txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        bot, [chatbot, lang_dropdown], chatbot
    )
    txt_msg.then(lambda: gr.Textbox(interactive=True),
                 None, [txt], queue=False)

demo.queue()
if __name__ == "__main__":
    demo.queue().launch(debug=True, share=True)
