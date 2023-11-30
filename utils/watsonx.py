from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods

# display example params
GenParams().get_example_values()

def get_watsonxai_response (
  input_text, instruction, max_new_tokens
):
  # model params
  generate_params = {
    GenParams.DECODING_METHOD: "greedy",
    GenParams.MIN_NEW_TOKENS: 50,
    GenParams.MAX_NEW_TOKENS: max_new_tokens,
    GenParams.REPETITION_PENALTY: 2
  }

  # model input
  model = Model(
    model_id = ModelTypes.GRANITE_13B_INSTRUCT,
    params = generate_params,
    credentials = {
      "apikey": "aoopT5yVxVAu6YCkRXqM3ubDXM3HzJh2lDvzDuaR2VqG",
      "url": "https://us-south.ml.cloud.ibm.com"
    },
    project_id="0ead8ec4-d137-4f9c-8956-50b0da4a7068"
  )

  # model invocation
  # generated_response = model.generate(prompt=input) # complete json output
  generated_response = model.generate_text(prompt=instruction + "\n\n" + input_text) # only generated text
  print(generated_response)
  return generated_response

# end of get_watsonxai_response

# test sample
# test_input = "You are an insurance claim processor. Read the following letter from an insurance customer and extract the requestor name, the requestor email, the date of the request, the policy number, the insurance plan name, and the reimbursement amount. Format the output as shown in the example.\n\nExample Input:\nInsurance Claim Letter for Reimbursement\nFrom: __Mike Hawthorne_________ Sender’s Name\n____1234 Main Street____________\n_________________________________ Sender’s Address\n____Los Angeles, CA 90045_________ City, State, ZIP Code\n____hawm1234@gmail.com_________ Sender’s email\n____September 21, 2023_____________ Date\nTo Whom It May Concern,\nTo:____UltraCo Insurance________ Recipient’s Name\n_______2093 Lakewood Drive______\n_________________________________ Recipient’s Address\n______Sayreville, NJ 08872________ City, State, ZIP Code\nMy name is __Mike Hawthorne_______ and I have an active medical insurance policy with your company. The policy number is _175032218_______. I am writing this letter to claim reimbursement for two fillings from Stanley Smith, DDS., who is an in-network provider for UltraCo Insurance.\nEnclosed with this letter are the following supporting documents for your reference and review:\nThe original bill for $540.25 and a copy of the UltraCo website page that lists Stanley Smith, DDS. as an in-network provider in the SpecialDentalPlus plan.\nAccording to section __13.1__ of the policy, I am entitled to __80__% reimbursement. Kindly acknowledge the receiving of this claim and advise whether I have to fill some form or whether you require other documents to complete the claim request.\nLooking forward to your response and cooperation.\nSincerely,\n__Mike Hawthorne_______________ Sender’s Name\n__Mike Hawthorne__________________ Sender’s Signature\n\nExample Output:\nname: Mike Hawthorne\nemail: hawm1234@gmail.com\ndate: September 28, 2023\npolicy number: 175032218\ninsurance plan name: SpecialDentalPlus\nreimbursement amount: $540.25\n\nInput:\nInsurance Claim Letter for Reimbursement\nFrom: __Susan Brown_________\nSender's Name\n____4030 Summer Street____________\n_________________________________\nSender's Address\n____Los Angeles, CA 90045_________\nCity, State, ZIP Code\n____sbrn1234@gmail.com_________\nSender's email\nTo:____UltraCo Insurance________\nRecipient's Name\n_______2093 Lakewood Drive______\n_________________________________\nRecipient's Address\n______Sayreville, NJ 08872________\nCity, State, ZIP Code\n____September 21, 2023_____________\nDate\nTo Whom It May Concern,\nMy name is __Susan Brown_______ and I have an active medical insurance policy with\nyour company. The policy number is _200032218_______. I am writing this letter to\nclaim reimbursement for $9.20 for a custom crown provided by Smiley Boutique,\nwho is an in-network provider for UltraCo Insurance.\nEnclosed with this letter are the following supporting documents for your reference and\nreview:\nI am including the previous EOB that UltraCo says under the SpecialDentalPlus plan\nthat Smiley Boutique is not a network provider. I am also including a copy of the\nUltraCo website page that lists Smiley Boutique as an in-network provider.\nAccording to section __15.2__ of the policy, I am entitled to __80__% reimbursement.\nKindly acknowledge the receiving of this claim and advise whether I have to fill some\nform or whether you require other documents to complete the claim request.\nLooking forward to your response and cooperation.\nSincerely,\n__Susan Brown_______________\nSender's Name\n__Susan Brown__________________\nSender's Signature\n\nOutput:\n"
# get_watsonxai_response(test_input)

'''
{
  "model_id": "ibm/granite-13b-instruct-v1",
  "input": "${inputs}",
  "parameters": {
    "decoding_method": "greedy",
    "max_new_tokens": 200,
    "min_new_tokens": 0,
    "stop_sequences": [],
    "repetition_penalty": 1
  },
  "project_id": "0ead8ec4-d137-4f9c-8956-50b0da4a7068"
}
'''
