# Text Summarization using IBM watsonx.ai
IBM watsonx AI offers generative AI capabilities powered by foundation models and traditional machine learning. In watsonx.ai, you have access to IBM selected open source models from Hugging Face, as well as other third-party models including Llama-2-chat and StarCoder LLM for code generation, and a family of IBM-trained foundation models of different sizes and architectures. These models start with Slate for non-generative AI tasks and the Granite series models that use a decoder architecture to support a variety of enterprise NLP generative AI tasks.

This repository demonstrates how to use [watson machine learning](https://ibm.github.io/watson-machine-learning-sdk/) Python SDK to call watsonx.ai models from Cloudera Machine Learning (CML) workspace. In this [Applied ML Prototype (AMP)](https://docs.cloudera.com/machine-learning/cloud/applied-ml-prototypes/topics/ml-amps-overview.html), text summarization based on custom instruction is used as an example but these foundation models are capable of much more such as question answering, classification, extraction and so on.

## AMP Overview
This AMP provides two model choices - ibm/granite-13b-chat-v1 and meta-llama/llama-2-70b-chat, along with text areas to provide custom instruction & input text and the ability to modify parameters for advanced generation.

![image](/assets/app_interface.png)

## AMP Prerequisites
- Access to CML [workspace](https://docs.cloudera.com/machine-learning/cloud/workspaces/topics/ml-provision-workspaces.html).
- [IBM Cloud account](https://www.ibm.com/cloud).
- In IBM cloud account, create API Key [here](https://cloud.ibm.com/iam/apikeys).
- Access to IBM Project. This is where foundation models are hosted.
 
 ## AMP Setup
 - Create a new CML project using AMPs setup option and provide `https://github.com/agupta-git/CML_AMP_watsonxai` GIT URL.
 - Provide values for required environment variables:
   - IBM_WATSONXAI_ENDPOINT - _this looks like `https://us-south.ml.cloud.ibm.com`_
   - IBM_API_KEY - _value of API Key you created in your IBM cloud account_
   - IBM_PROJECT_ID - _Id of the IBM project that you have access to_
- ML Runtime - PBJ Workbench, Python 3.9, Standard Edition and 2023.08 Version
- Launch the project. It takes about 3 minutes to execute the two steps. Once they are done, open the application.
- Click on the `Forest` example to load sample text in the input text area.

Enjoy!
