import os
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods

# method - get_watsonxai_response
def get_watsonxai_response (
    modelId, input_text, custom_instruction, max_new_tokens, temperature, top_p
):
    # model params
    generate_params = {
        GenParams.DECODING_METHOD: DecodingMethods.SAMPLE,
        GenParams.MIN_NEW_TOKENS: 0,
        GenParams.MAX_NEW_TOKENS: max_new_tokens,
        GenParams.REPETITION_PENALTY: 1,
        GenParams.TEMPERATURE: temperature,
        GenParams.TOP_P: top_p,
        GenParams.TOP_K: 50
    }

    # model choices
    model_id_str = ""
    if modelId == "ibm/granite-13b-chat-v1":
        model_id_str = ModelTypes.GRANITE_13B_CHAT
    elif modelId == "meta-llama/llama-2-70b-chat":
        model_id_str = ModelTypes.LLAMA_2_70B_CHAT

    # capture environment variables
    env_ibm_api_key = os.environ.get("IBM_API_KEY")
    env_ibm_watsonxai_endpoint = os.environ.get("IBM_WATSONXAI_ENDPOINT")
    env_ibm_project_id = os.environ.get("IBM_PROJECT_ID")
    print("Environment Variables:"
            + "\nIBM_API_KEY: " + env_ibm_api_key
            + "\nIBM_WATSONXAI_ENDPOINT: " + env_ibm_watsonxai_endpoint
            + "\nIBM_PROJECT_ID: " + env_ibm_project_id
    )

    # prepare model input
    model = Model(
        model_id = model_id_str,
        params = generate_params,
        credentials = {
            "apikey": env_ibm_api_key,
            "url": env_ibm_watsonxai_endpoint
        },
        project_id = env_ibm_project_id
    )

    # model invocation
    input_prompt = custom_instruction + "\nInput: \n" + input_text + "\n\n Output:"
    print("Input---\n" + input_prompt)
    generated_response = model.generate_text(prompt=input_prompt)
    return generated_response
# end of get_watsonxai_response

# test sample
model_choice = "ibm/granite-13b-chat-v1" # options: ibm/granite-13b-chat-v1, meta-llama/llama-2-70b-chat
custom_instruction = "Please provide a summary of the following text. Do not add any information that is not mentioned in the text below."
input_text = "A large language model (LLM) is a type of language model notable for its ability to achieve general-purpose language understanding and generation. LLMs acquire these abilities by using massive amounts of data to learn billions of parameters during training and consuming large computational resources during their training and operation. LLMs are artificial neural networks (mainly transformers) and are (pre-)trained using self-supervised learning and semi-supervised learning.\n As autoregressive language models, they work by taking an input text and repeatedly predicting the next token or word. Up to 2020, fine tuning was the only way a model could be adapted to be able to accomplish specific tasks. Larger sized models, such as GPT-3, however, can be prompt-engineered to achieve similar results. They are thought to acquire embodied knowledge about syntax, semantics and ontology inherent in human language corpora, but also inaccuracies and biases present in the corpora. \n Notable examples include OpenAI's GPT models (e.g., GPT-3.5 and GPT-4, used in ChatGPT), Google's PaLM (used in Bard), and Meta's LLaMa, as well as BLOOM, Ernie 3.0 Titan, and Anthropic's Claude 2."
get_watsonxai_response(model_choice, input_text, custom_instruction, 512, 0.7, 1)
