from dotenv import load_dotenv
import os
import gradio as gr

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")

system_prompt = """
You are Alfred, a composed and analytical AI assistant.

You speak with calm confidence and subtle refinement.
Not dramatic, not fictional — just sharp and slightly polished.

Tone:
- Intelligent
- Slightly dry humor
- Minimal but elegant phrasing
- Professional but not cold

When appropriate:
- Offer insight beyond the surface
- Add brief reflective observations
- Encourage deeper thinking

Never:
- Overuse humor

Your presence should feel as if speaking with a sharp, thoughtful mentor.
"""


llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    google_api_key=gemini_key,
    temperature=0.4
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    (MessagesPlaceholder(variable_name="history")),
    ("user", "{input}")]
)


chain = prompt | llm | StrOutputParser()


def chat(user_in, hist):
    langchain_history = []

    for item in hist:
        if item["role"] == "user":
            langchain_history.append(HumanMessage(content=item["content"]))
        elif item["role"] == "assistant":
            langchain_history.append(AIMessage(content=item["content"]))

    response = chain.invoke({"input": user_in, "history": langchain_history})

    return "", hist + [{"role": "user", "content": user_in},
                       {"role": "assistant", "content": response}]

def clear_chat():
    return "", []

page = gr.Blocks(
        title="Chat with Alfred",
        fill_height=True
)

with page:
    gr.Markdown(
        """
        Chat with Alfred\n
        Welcome to your personal conversation with Alfred.
        """
    )

    chatbot = gr.Chatbot(height=600)

    msg = gr.Textbox(placeholder="Ask Alfred anything...")

    msg.submit(chat, [msg, chatbot], [msg, chatbot])

    clear = gr.Button("Clear Chat")
    clear.click(clear_chat, outputs=[msg, chatbot])

page.launch(theme=gr.themes.Soft(), share=True)