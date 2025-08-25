import os
import json
from typing import Optional, List
from pydantic import BaseModel, Field
import openai


# Set your OpenAI API key (or use openai.api_key = "sk-..." directly):
key = "YOUR KEY"
client = openai.OpenAI(api_key=key)

# ---------------------------
# 1) Define possible values
# ---------------------------
KATEGORIE_VALUES = [
    "Academic/Research Awards",
    "Community & Public Engagement Awards",
    "No Award",
    "Professional [Field] Awards"
]

OBERBEGRIFF_VALUES = [
    "Awards from scientific societies",
    "Career Awards",
    "Grants",
    "International Collaboration Awards",
    "Publications",
    "Talks & Presentations",
    "Teaching",
    "Diversity & Inclusion"
]

UNTERBEGRIFF_1_VALUES = [
    "Early Career Awards",
    "Further Career Awards",
    "Fellowships & Scholarships"
]

UNTERBEGRIFF_2_VALUES = [
    "PhD",
    "Further Early Career Awards",
    "Habilitation"
]


# -------------------------------------------------------
# 2) Create a Pydantic model to validate each output row
# -------------------------------------------------------
class AwardPrediction(BaseModel):
    """
    Each instance of AwardPrediction represents the classification
    of a single award string into Kategorie, Oberbegriff, Unterbegriff_1, and Unterbegriff_2.
    """
    
    Kategorie: str = Field(
        ...,
        description=f"One of the values in {KATEGORIE_VALUES}."
    )
    Oberbegriff: Optional[str] = Field(
        None,
        description=f"One of the values in {OBERBEGRIFF_VALUES} or 'NaN'."
    )
    Unterbegriff_1: Optional[str] = Field(
        None,
        description=f"One of the values in {UNTERBEGRIFF_1_VALUES} or 'NaN'."
    )
    Unterbegriff_2: Optional[str] = Field(
        None,
        description=f"One of the values in {UNTERBEGRIFF_2_VALUES} or 'NaN'."
    )

df_string = '' # your dataframe example to be used as a reference for structuring 
# ---------------------------------
# 3) System prompt with CSV context
# ---------------------------------
CSV_EXAMPLE_TEXT = f"""
[STRING FORMATTED PANDAS DATAFRAME AS AN EXAMPLE]

{df_string}

Remember the valid sets:
Kategorie: [Academic/Research Awards, Community & Public Engagement Awards, No Award, Professional [Field] Awards]
Oberbegriff: [Awards from scientific societies, Career Awards, Grants, International Collaboration Awards, Publications, Talks & Presentations, Teaching, Diversity & Inclusion] or None
Unterbegriff_1: [Early Career Awards, Further Career Awards, Fellowships & Scholarships] or None
Unterbegriff_2: [PhD, Further Early Career Awards, Habilitation] or None

Always return valid JSON with these 4 keys:
{{
  
  "Kategorie": "...",
  "Oberbegriff": "...",
  "Unterbegriff_1": "...",
  "Unterbegriff_2": "..."
}}
If no category fits, use None for those fields except Kategorie which you always have to assign a value a to it. Use internet if you were not sure about an award. 
"""

SYSTEM_PROMPT = f"""
You are a classification model. 
You receive a short award string and must classify it into:
- Kategorie (one of the 4 valid options)
- Oberbegriff (one of the 8 valid options or NaN)
- Unterbegriff_1 (one of the 3 valid options or NaN)
- Unterbegriff_2 (one of the 3 valid options or NaN)

Here is the CSV excerpt illustrating how certain awards were categorized:

{CSV_EXAMPLE_TEXT}

Your goal: For any new input string, return valid JSON with exactly these keys:
  "Kategorie", "Oberbegriff", "Unterbegriff_1", "Unterbegriff_2".
"""


def classify_award_string(text: str,model) -> AwardPrediction:
    """
    Sends one input string to GPT and asks it to produce JSON
    that can be parsed as AwardPrediction. 
    """
    user_message = f"Classify this list of strings:\n{text}\n Return valid JSON for each string in the list."

    response = client.beta.chat.completions.parse(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ], 
        response_format = AwardPrediction
    )

    content = response.choices[0].message.parsed

    return content

    


def classify_award_list(strings: List[str],model) -> List[AwardPrediction]:
    """
    Classifies a list of award strings by calling GPT for each.
    Returns a list of AwardPrediction objects.
    """
    results = {}
    for s in strings:
        prediction = classify_award_string(text = s,model = model)
        results[s] = prediction.model_dump()
    return results


r = 50 
awards = [] # your list of awards
for i in range(0,len(awards),r): 
    predictions_4o = classify_award_list(awards[i:i+r],model="gpt-4o")


    # Path to the existing JSON file
    file_path = 'awards_structure.json'

    # Step 1: Open and load the existing JSON file
    with open(file_path, 'r', encoding='utf-8') as file:
        current_data = json.load(file)

    # Step 2: Update the current dictionary with the new data
    current_data.update(predictions_4o)

    # Step 3: Save the updated dictionary back to the JSON file
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(current_data, file, ensure_ascii=False, indent=4)

    print(f"New data appended successfully! chunk : {i}")

