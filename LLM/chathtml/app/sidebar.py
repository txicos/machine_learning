from os import getenv
import requests
import gettext
import logging
import streamlit as st

gxt = gettext.gettext
language = ''

class Language:
   
  def __init__(self) -> None:
      self._idiom = ''
      
  @property
  def idiom(self):
    return self._idiom
  
  @idiom.setter
  def idiom(self, value):
    self._idiom = value   


class Key:
  
  def __init__(self) -> None:
      self.open_api_key_input = None
  
  def set_open_api_key(self, api_key: str):
      self.open_api_key_input = api_key

  def get_open_api_key(self, ):
      return self.open_api_key_input
    
class LLMModel:
  
  def __init__(self) -> None:
      self._provider = None
      self._model = None
      
  @property
  def provider(self):
    return self._provider
  
  @provider.setter
  def provider(self, value):
    self._provider = value
    
  @property
  def model(self):
    return self._model
  
  @model.setter
  def model(self, value):
    self._model = value
    
def get_i18(message):
  #logging.debug(f'get {message} {gxt(message)}')
  return gxt(message)

def sidebar(key, llm_model, lang):
    with st.sidebar:
      
      language = st.sidebar.radio('', ['en_US', 'pt_BR'])
      if language == 'pt_BR':
          lang.idiom = 'pt_BR'
      else:
          lang.idiom = 'en_US'

      try:
        logging.debug(language)
        localizator = gettext.translation('chathtml', localedir='locale', languages=[language])
        localizator.install()
        global gxt
        gxt = localizator.gettext
        #st.rerun()
      except:
          pass
  
      open_api_key_input = key.get_open_api_key()
      
      selected_model = None
      model_provider =  None
      
      model_provider = st.selectbox(
          gxt("ai_provider"), #Select your preferred model provider:
          ["OpenAI", "Ollama"],
          key="model_provider",
          help=gxt("help_select_model") #"Select the model provider you would like to use. This will determine the models available for selection.",
      )
      
      if model_provider == "OpenAI":                
        how_to_use = gxt("how_to_use")
        # "## How to use\n"
        # "Enter your [OpenAI API key](https://platform.openai.com/account/api-keys) below🔑\n"  
        if open_api_key_input is None:
            st.markdown(
            how_to_use
            )
            get_openai_api_key = gxt("get_openai_key")
            input_openai_key = gxt("input_key")
            open_api_key_input = st.text_input(
                input_openai_key,
                type="password",
                placeholder=gxt("paste_openai_key"),#"Paste your API key here (sk-...)"
                help=f"{get_openai_api_key} https://platform.openai.com/account/api-keys.",  #You can get your API key from
            )

            if len(open_api_key_input)>0:
                key.set_open_api_key(open_api_key_input)
            else:
                st.error(gxt("configure_openai_key_error")) #"Please configure your Open API key!"

        else:
            st.markdown(gxt("configure_openai_ok")) #"Open API Key Configured!"
            
        selected_model = st.selectbox(
          gxt("ai_model"),#"Select the model you would like to use:",
          ["gpt-4o", "gpt-4o-mini"],
          key="selected_model",
          help= gxt("openai_latest")#"GPT-4o and GPT-4o mini"
          # curl https://api.openai.com/v1/models \
          # -H "Authorization: Bearer $OPEN_API_KEY"
      )
          
      elif model_provider == "Ollama":
            # Make a request to the Ollama API to get the list of available models
            try:
                host = getenv('OLLAMA_HOST')
                url = host + "/api/tags"
                response = requests.get(url)
                response.raise_for_status() # Raise an exception for 4xx/5xx status codes
            except requests.exceptions.RequestException as e:
                st.error(gxt("ollama_end_point_notfound")) #"Ollama endpoint not found, please select a different model provider."
                response = None
            
            if response:
                data = response.json()
                available_models = [model["name"] for model in data["models"]]
                # Add model selection input field to the sidebar
                selected_model = st.selectbox(
                    gxt("ollama_select_model"), #"Select the model you would like to use:"
                    available_models,
                    key="selected_model",
                )
                  
      llm_model.provider = model_provider
      llm_model.model = selected_model
