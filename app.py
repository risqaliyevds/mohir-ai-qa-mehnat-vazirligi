import gradio as gr
import time
from config import *
from retriever import retrieverQA
from main import correct_unicode, translateQuestion
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain


def add_text(history, text):
    history = history + [(text, None)]
    return history, text

def bot(history, lang_dropdown):
    response = history[-1][0]
    history[-1][1] = ""

    llm = ChatOpenAI(
        model_name="gpt-4",
        temperature=0,
        openai_api_key=os.environ['OPENAI_API_KEY'])

    qa_chain = load_qa_chain(llm, chain_type="stuff")

    if lang_dropdown == "UZ":
        question_uz = response
        question_en = translateQuestion(
            question_uz, source_language="uz", target_language='en')
        matched_docs = retrieverQA.get_relevant_documents(question_en)
        answer_en = qa_chain.run(input_documents = matched_docs, question = question_en)
        answer_uz = correct_unicode(
            translateQuestion(answer_en, source_language="en", target_language='uz'))
        response = answer_uz
    else:
        matched_docs = retrieverQA.get_relevant_documents(response)
        answer_en = qa_chain.run(input_documents = matched_docs, question = response)
        response = answer_en

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
            None, (os.path.join(os.path.dirname(__file__), os.environ["BOT_LOGO_PATH"])),
    ))

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
