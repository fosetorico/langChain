# from langchain.llms import OpenAI
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from secret_key import openapi_key

import os
os.environ['OPENAI_API_KEY'] = openapi_key

llm = OpenAI(temperature=0.7)

# def generate_restaurant_name_and_items(cuisine):
#     # Chain 1: Restaurant Name
#     prompt_template_name = PromptTemplate(
#         input_variables=['cuisine'],
#         template="I want to open a restaurant for {cuisine} food. Suggest a fancy name for this."
#     )
#
#     name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="restaurant_name")
#     # name_chain = prompt(llm=llm, prompt=prompt_template_name, output_key="restaurant_name")
#
#     # Chain 2: Menu Items
#     prompt_template_items = PromptTemplate(
#         input_variables=['restaurant_name'],
#         template="""Suggest some menu items for {restaurant_name}. Return it as a comma separated string"""
#     )
#
#     food_items_chain = LLMChain(llm=llm, prompt=prompt_template_items, output_key="menu_items")
#     # food_items_chain = prompt(llm=llm, prompt=prompt_template_items, output_key="menu_items")
#
#     chain = SequentialChain(
#         chains=[name_chain, food_items_chain],
#         input_variables=['cuisine'],
#         output_variables=['restaurant_name', "menu_items"]
#     )
#
#     response = chain({'cuisine': cuisine})
#
#     return response

def generate_restaurant_name_and_items(cuisine):
    # Define the prompt template for generating the restaurant name
    prompt_template_name = PromptTemplate.from_template(
        "I want to open a restaurant for {cuisine} food. Suggest a fancy name for this."
    )

    # Generate the restaurant name
    name_prompt = prompt_template_name.format(cuisine=cuisine)
    restaurant_name = llm.invoke(name_prompt)

    # Define the prompt template for generating the menu items
    prompt_template_items = PromptTemplate.from_template(
        "Suggest some menu items for {restaurant_name}. Return it as a comma separated string."
    )

    # Generate the menu items
    items_prompt = prompt_template_items.format(restaurant_name=restaurant_name)
    menu_items = llm.invoke(items_prompt)

    return {
        "restaurant_name": restaurant_name,
        "menu_items": menu_items
    }

if __name__ == "__main__":
    print(generate_restaurant_name_and_items("Nigerian"))
