
import streamlit as st

import logging

from sidebar import sidebar, Key, LLMModel, get_i18, Language

from langchain_community.chat_models import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.schema import (
    HumanMessage,
    SystemMessage
)

from scrap import get_text_from_url, perform_rag

logging.getLogger().setLevel(logging.INFO)

def get_chat_object(llm_model, api_key=None):
  if llm_model.provider == "OpenAI" and api_key is not None:
    chat = ChatOpenAI(temperature=0.1, model_name=llm_model.model, openai_api_key=api_key) 
  elif llm_model.provider == "Ollama":
    chat = ChatOllama(temperature=0.1, model=llm_model.model) 
  else: 
    chat = None
  return chat

def summarize_content(content, chat, lang):
    """
    Summarizes the content to fit within token limits.
    """
    try:
        messages = [
            SystemMessage(content="Você é um assistente que resume textos."),
            HumanMessage(content= f"Escreva um resumo usando a língua {lang.idiom} para o seguinte texto:\n\n{content}")
        ]

        response = chat(messages)
        summary = response.content.strip()
        return summary
    except Exception as e:
        logging.error(f"Error during summarization: {e}")
        return None

def ask_question_about_content(content_txt, question, chat, lang):
    """
    Asks a question based on the provided content.
    """
    try:

        messages = [
            SystemMessage(content= "Você é um assistente prestativo capaz de responder perguntas \
                                    sobre um assunto tratado em um texto que te será apresentado."),
            HumanMessage(content= f"""Baseado apenas no seguinte texto: \n"{content_txt}"\n, 
            providencie uma resposta para uma pergunta que será feita mais adiante, mencionando  
            quais partes do conteúdo do texto você usou para elaborar a sua resposta. Devolva 
            um texto bem elaborado, mas direto. Não faça divagações. Segue a Pergunta:\n\n{question}""")
        ]
        

        response = chat(messages)
        answer = response.content.strip()
        logging.info(f"Translating: {answer}")
        messages = [
            SystemMessage(content= "Você é um tradutor de textos."),
            HumanMessage(content= f"""Traduza o texto \n"{answer}"\n.
            para o idioma {lang.idiom}. Se estiver no mesmo idioma, apenas repita o texto.""")
        ]
        
        response = chat(messages)
        answer = response.content.strip()
        logging.info(f"Texto final: {answer}")
        return answer
    except Exception as e:
        logging.error(f"Error during question processing: {e}")
        return None

def user_intent(question, chat):
    try:

        messages = [
            SystemMessage(content= "Você detectará a intenção de uma pergunta."),
            HumanMessage(content= f"""Dada a seguinte pergunta: \n {question}.  \n 
                         Emita uma afirmação dizendo se a pergunta indica a 
                         intenção de resumir um assunto ou indagar algo sobre o 
                         assunto. Sua resposta deve ser apenas a palavra resumir 
                         ou a indagar. Não acrescente mais texto ou faça divagações!""")
        ]

        logging.info(f"verifica intencao  ++++++++++++++++++++++++++++++++++++ ")
        response = chat(messages)
        answer = response.content.strip()
        logging.info(f"intencao  ++++++++++++++++++++++++++++++++++++ {answer}")
        if answer.lower().find("resumir")>=0:
          return True
        return False
    except Exception as e:
        logging.error(f"Error during question processing: {e}")
        return False


st.set_page_config(
    layout="centered",
    initial_sidebar_state="auto",
)


logging.getLogger().setLevel(logging.INFO)
raw_content = ''
# these containers appear in the order they are declared
urlcontainer = st.container()
#container for the user's text input
usercontainer = st.container()

def app():
    
    if 'chat' not in st.session_state:
      st.session_state.chat = None

    if 'url' not in st.session_state:
      st.session_state.url = None


    global raw_content
    key = Key()
    
    llm_model = LLMModel()

    lang = Language()
    
    sidebar(key, llm_model, lang)
    #logging.info(f"{llm_model.provider}  {llm_model.model}")

    if llm_model.provider is not None and llm_model.model is not None:
      
      st.session_state.chat = get_chat_object(llm_model, key.get_open_api_key())
      
      if st.session_state.chat is not None:
      
        with urlcontainer:
          
          st.title(get_i18("urlcontentqa")) #URL Content Q&A

          # Input for URL
          st.session_state.url = st.text_input(get_i18('enterurl')) #Enter the URL:
          logging.info(f"URL: {st.session_state.url}")
          if st.session_state.url and len(raw_content) == 0:  # Check if URL is provided
            
            with st.spinner(get_i18('fetchurlcontent')): #Fetching content from the URL...
                raw_content = get_text_from_url(st.session_state.url)
                logging.debug(f"{raw_content}")
                if len(raw_content) == 0:
                  st.error(get_i18('failedretrievecontent'))

          if st.session_state.url and len(raw_content) > 0:
            with usercontainer:
                logging.info("Ready for questions")
                with st.form(key='my_form', clear_on_submit=True):
                  
                    talk_about_faq = get_i18("talk_about_faq")
                    #"Talk about your faq here (:"
                    user_input = st.text_input("Query:", placeholder=talk_about_faq, key='input')
                    
                    content = raw_content
                    
                    if user_input:

                        summarize = user_intent(user_input, st.session_state.chat)
                        if (not summarize) and llm_model.provider == 'Ollama':
                          content = perform_rag(raw_content,user_input)
                          
                          if content is None:
                              st.error(get_i18('failedsummarizecontent')) #Failed to summarize content.
                          else:
                              st.success(get_i18('contentsummarized')) # Content

                    submited = st.form_submit_button(label='Send')
                    if submited and user_input:
                      
                      with st.spinner(get_i18('processingquestion')): #Processing your question...
                        logging.info(f"new question received {user_input} sobre conteudo {len(content)} idioma {lang.idiom}")
                        #logging.debug(f"{content}")
                        if summarize:
                            answer = summarize_content(content, st.session_state.chat, lang)
                        else:
                          answer = ask_question_about_content(content, user_input, st.session_state.chat, lang)
                        logging.info(f"answer {answer} received ")
                      if answer:
                        st.write(user_input)
                        st.write(get_i18('answer'), answer) #Answer:
                      else:
                        st.error(get_i18('failedgetanswer')) #Failed to get an answer.


                col1, _ = st.columns([1, 3.2])
                if col1.button(get_i18("clearhistory")):
                              
                    st.rerun()
                    st.session_state.chat = None
                    raw_content = ''

      else:
        st.title(get_i18("HTMLChatBot"))
        st.subheader(get_i18("askdocument"))        

        st.error(get_i18("configureprovidermodel")) #Please configure provider and model
        #raw_content = ''


                            

def main() -> None:

    app()


if __name__ == "__main__":
    
    main()        

                
#streamlit run tuto_chatbot_csv.py