import streamlit as st
from huggingface_hub import InferenceClient
import os
import sys

st.title("StrangerX AI")

base_url="https://api-inference.huggingface.co/models/"

API_KEY = os.environ.get('HUGGINGFACE_API_KEY')
# print(API_KEY)
# headers = {"Authorization":"Bearer "+API_KEY}

model_links ={
    "Mistral-7B":base_url+"mistralai/Mistral-7B-Instruct-v0.2",
    "Mistral-22B":base_url+"mistral-community/Mixtral-8x22B-v0.1",
    "Phi-3":base_url+"microsoft/Phi-3-mini-4k-instruct"
}

#Pull info about the model to display
model_info ={
    "Mistral-7B":
        {'description':"""The Mistral model is a **Large Language Model (LLM)** that's able to have question and answer interactions.\n \
            \nIt was created by the [**Mistral AI**](https://mistral.ai/news/announcing-mistral-7b/) team as has over  **7 billion parameters.** \n""",
        'logo':'https://mistral.ai/images/logo_hubc88c4ece131b91c7cb753f40e9e1cc5_2589_256x0_resize_q97_h2_lanczos_3.webp'},

    
    "Mistral-22B":
        {'description':"""The Mistral model is a **Large Language Model (LLM)** that's able to have question and answer interactions.\n \
            \nIt was created by the [**Mistral AI**](https://mistral.ai/news/announcing-mistral-22b/) team as has over  **22 billion parameters.** \n""",
        'logo':'https://mistral.ai/images/logo_hubc88c4ece131b91c7cb753f40e9e1cc5_2589_256x0_resize_q97_h2_lanczos_3.webp'},

    
      "Phi-3":        
      {'description':"""The PHI 3 model is a **Large Language Model (LLM)** that's able to have question and answer interactions.\n \
          \nIt was created by the [**Microsoft Team**](https://news.microsoft.com/source/features/ai/the-phi-3-small-language-models-with-big-potential/) team as has over  **< 13 billion parameters.** \n""",
      'logo':'https://www.techfinitive.com/wp-content/uploads/2023/07/microsoft-365-copilot-jpg.webp'},

    
    
    # "Zephyr-7B-β":        
    # {'description':"""The Zephyr model is a **Large Language Model (LLM)** that's able to have question and answer interactions.\n \
    #     \nFrom Huggingface: \n\
    #     Zephyr is a series of language models that are trained to act as helpful assistants. \
    #     [Zephyr-7B-β](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta)\
    #     is the second model in the series, and is a fine-tuned version of mistralai/Mistral-7B-v0.1 \
    #     that was trained on on a mix of publicly available, synthetic datasets using Direct Preference Optimization (DPO)\n""",
    # 'logo':'https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha/resolve/main/thumbnail.png'},

}

def format_promt(message, custom_instructions=None):
    prompt = ""
    if custom_instructions:
        prompt += f"[INST] {custom_instructions} [/INST]"
    prompt += f"[INST] {message} [/INST]"
    return prompt

def reset_conversation():
    '''
    Resets Conversation
    '''
    st.session_state.conversation = []
    st.session_state.messages = []
    return None

models =[key for key in model_links.keys()]

# Create the sidebar with the dropdown for model selection
selected_model = st.sidebar.selectbox("Select Model", models)

#Create a temperature slider
temp_values = st.sidebar.slider('Select a temperature value', 0.0, 1.0, (0.5))

#Add reset button to clear conversation
st.sidebar.button('Reset Chat', on_click=reset_conversation) #Reset button

# Create model description
st.sidebar.write(f"You're now chatting with **{selected_model}**")
st.sidebar.markdown(model_info[selected_model]['description'])
st.sidebar.image(model_info[selected_model]['logo'])
st.sidebar.markdown("*Generated content may be inaccurate or false.*")
st.sidebar.markdown("\nAbout the Developer of StrangerX AI[here](https://github.com/PRITHIVSAKTHIUR/StrangerX).")

if "prev_option" not in st.session_state:
    st.session_state.prev_option = selected_model

if st.session_state.prev_option != selected_model:
    st.session_state.messages = []
    # st.write(f"Changed to {selected_model}")
    st.session_state.prev_option = selected_model
    reset_conversation()

#Pull in the model we want to use
repo_id = model_links[selected_model]

st.subheader(f'{selected_model}')
# st.title(f'ChatBot Using {selected_model}')

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Accept user input
if prompt := st.chat_input(f"Hi I'm {selected_model}🗞️, How can I help you today?"):

    custom_instruction = "Act like a Human in conversation"

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    formated_text = format_promt(prompt, custom_instruction)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        client = InferenceClient(
            model=model_links[selected_model],)
            # headers=headers)

        output = client.text_generation(
            formated_text,
            temperature=temp_values,#0.5
            max_new_tokens=3000,
            stream=True
        )

        response = st.write_stream(output)
    st.session_state.messages.append({"role": "assistant", "content": response})
