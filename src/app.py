import gradio as gr
from utils.chatbot import ChatBot
from utils.ui_settings import UISettings

# Instantiate the ChatBot class
chatbot_instance = ChatBot()

with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("RAG with Cancer Data"):
            ##############
            # First ROW:
            ##############
            with gr.Row() as row_one:
                chatbot = gr.Chatbot(
                    [],
                    elem_id="chatbot",
                    bubble_full_width=False,
                    height=500,
                    avatar_images=(
                        ("images/AI_RT.png"), "images/openai.png")
                )
                # **Adding like/dislike icons
                chatbot.like(UISettings.feedback, None, None)
            ##############
            # SECOND ROW:
            ##############
            with gr.Row():
                input_txt = gr.Textbox(
                    lines=4,
                    scale=8,
                    placeholder="Ask questions about the cancer dataset",
                    container=False,
                )
            ##############
            # Third ROW:
            ##############
            with gr.Row() as row_two:
                text_submit_btn = gr.Button(value="Submit")
                clear_button = gr.ClearButton([input_txt, chatbot])

            ##############
            # Process:
            ##############
            txt_msg = input_txt.submit(
                fn=chatbot_instance.respond,  # Use the instance method
                inputs=[
                    chatbot, 
                    input_txt,
                    gr.State("RAG with stored CSV/XLSX ChromaDB"),  # Fixed chat_type
                    gr.State("Chat")  # Fixed app_functionality
                ],
                outputs=[input_txt, chatbot],
                queue=False
            ).then(
                lambda: gr.Textbox(interactive=True),
                None,
                [input_txt],
                queue=False
            )

            txt_msg = text_submit_btn.click(
                fn=chatbot_instance.respond,  # Use the instance method
                inputs=[
                    chatbot, 
                    input_txt,
                    gr.State("RAG with stored CSV/XLSX ChromaDB"),  # Fixed chat_type
                    gr.State("Chat")  # Fixed app_functionality
                ],
                outputs=[input_txt, chatbot],
                queue=False
            ).then(
                lambda: gr.Textbox(interactive=True),
                None,
                [input_txt],
                queue=False
            )

if __name__ == "__main__":
    demo.launch()