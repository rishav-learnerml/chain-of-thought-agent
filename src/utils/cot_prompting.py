from src.llm.llm import LLM
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from src.schema.output_schema import OutputSchema
from typing import cast
import os
import requests

def get_weather_info(city: str) -> str:
    """
    Function to get weather information for a given city.

    Args:
        city (str): Name of the city to get weather information for.

    Returns:
        str: A string containing the weather information.
    """

    url=f"https://wttr.in/{city.lower()}?format=%C+%t+%w"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return f"The current weather in {city} is: {response.text}"
        else:
            return "Could not retrieve weather information at this time."
    except Exception as e:
        return "An error occurred while fetching weather information."
    


def cot_prompting(user_query: str)->None:
    print("Starting CoT Prompting...\n\n\n")

    system_prompt = SystemMessage(content=f"""You are a helpful AI assistant who can resolve user queries using the Chain of Thought. You will be given a user query and you will need to resolve it using the Chain of Thought.
                                  
    You will solve the user query by START, PLAN and OUTPUT.
    You can also call a tool if required from the available list of tools during the PLAN step.

    START: You will start by thinking about the user query and what the user is asking for.
    PLAN: You will then plan how to solve the user query. This step can be repeated multiple times until you are confident in your plan.
    TOOL: call the necessary tool with proper input
    OUTPUT: You will then execute the plan.

    Rules:
                                  
    - Always follow the JSON Output Format provided below.
    - Always use the PLAN step to break down the problem into smaller steps.
    - Always use the PLAN step to think about what information you need to solve the step.
    - Always use the PLAN step to think about what tools you can use to solve the problem.
    - Do not skip any step.
    - Do not get stuck in on step.
    - The sequence of steps should always be START (Where user gives an input) -> PLAN (multiple times) -> TOOL (if needed/optional) -> OUTPUT (Final response).

    Output Format:
    {{
        "step": "START" | "PLAN" | "OUTPUT" | "TOOL",
        "content": "string",
        "tool":dict(
            "name": "tool_name",
            "input": {{"param1":"input1","param2":"input2"}}
        ) | None
    }}
                                  
    Available Tools:

    - get_weather_info(city: str) -> str: A function that takes in a city name as input and returns the current weather information for that city.

    Examples:

    Q. What is the capital of France?
    A.
    {{"step": "START", "content": "What is the capital of France?"}}
    {{"step": "PLAN", "content": "Alright, Seems like the user is asking for the capital of France. So we need to find the capital of France."}}
    {{"step": "OUTPUT", "content": "The capital of France is Paris."}}

    Q. What is 23+15-12?
    A.
    {{"step": "START", "content": "What is 23+15-12?"}}
    {{"step": "PLAN", "content": "Alright, Seems like the user is asking for the result of 23+15-12. So we need to find the result of 23+15-12."}}
    {{"step": "PLAN", "content": "I will use the BODMAS rule to solve the problem."}}
    {{"step": "PLAN", "content": "BODMAS stands for Brackets, Orders, Division, Multiplication, Addition, and Subtraction."}}
    {{"step": "PLAN", "content": "So we need to solve the problem in the following order: 23+15-12."}}
    {{"step": "PLAN", "content": "First we will solve 23+15."}}
    {{"step": "PLAN", "content": "The output of 23+15 is 38."}}
    {{"step": "PLAN", "content": "Then we will solve 38-12."}}
    {{"step": "PLAN", "content": "The output of 38-12 is 26."}}
    {{"step": "OUTPUT", "content": "The result of 23+15-12 is 26."}}
                                  
    Q. What is the weather in New York?
    A.
    {{"step": "START", "content": "What is the weather in New York?"}}
    {{"step": "PLAN", "content": "The user is asking for the current weather in New York. I will need to use a tool to fetch the current weather information."}}
    {{"step": "TOOL", "content": "string", "tool": {{"name":"get_weather_info","input":{{"city":"New York"}}}}}}
    {{"step": "OUTPUT", "content": "The current weather in New York is: Partly cloudy +22°C ↗5km/h"}}
    """)

    message_history = [
        system_prompt,
        HumanMessage(content=user_query),
    ]

    print("🤔 ",user_query)
    llm = LLM(model_name=str(os.getenv("GROQ_MODEL_NAME","")))
    model = llm.getLLM().with_structured_output(OutputSchema)

    max_iterations = 20
    iteration = 0

    while iteration < max_iterations:           
        raw = model.invoke(message_history)
        response = cast(OutputSchema, raw)
        # Handle both dict and BaseModel response
        step = getattr(response, 'step', None)
        content = getattr(response, 'content', None)
        if step is None or content is None:
            if isinstance(response, dict):
                step = response.get('step')
                content = response.get('content')
        message_history.append(AIMessage(content=f"{{'step': '{step}', 'content': '{content}'}}"))

        if step == "START":
            print("🚀 ", response)

        elif step == "PLAN":
            print("🧠 ", response)

        elif step == "TOOL":
            print("🔧 ", response)
            if response.tool:
                tool_name = response.tool.name
                tool_input = response.tool.input
                if tool_name == "get_weather_info":
                    city = tool_input.get("city","")
                    if city:
                        tool_output = get_weather_info(city)
                        message_history.append(AIMessage(content=f"Tool Output: {tool_output}"))
                        print("🛠️  Tool executed. Output added to the conversation.")
                    else:
                        print("⚠️  Tool input missing 'city' parameter.")
                else:
                    print(f"⚠️  Unknown tool: {tool_name}")
            else:
                print("⚠️  No tool information provided.")

        elif step == "OUTPUT":
            print("🤖 ", content)
            break
        else:
            print(f"⚠️  Unexpected step: {step}. Exiting loop.")
            break
        iteration += 1
        
    else:
        print("⚠️  Max iterations reached. Exiting loop.")


    print("\n\n\nFinished CoT Prompting.")





    

