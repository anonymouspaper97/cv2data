from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import openai
import pandas as pd 
from langchain.llms import OpenAI
import os 

from utils.dbtools import connect_to_database
import json 

import utils.web_scraper 
import time 


key = "YOUR KEY"
os.environ["OPENAI_API_KEY"] = key


llm = ChatOpenAI(
     model_name="gpt-4-turbo-preview",
    temperature=0,
)

embedding=OpenAIEmbeddings() 



ALL = True # we assume, all the information is relevant so we do not use vector databases
verbose = True # to see the complete prumpt sent to ChatGPT set this to true 
###############################################################

conn = connect_to_database()
cursor = conn.cursor()

query = f"SELECT prof_id FROM webpage_content WHERE extracted_cv!= '' AND extracted_cv IS NOT NULL"

cursor.execute(query)

# Fetch the results
rows = cursor.fetchall()

ids = [row[0] for row in rows]

ids = sorted(list(set(ids)))
cursor.close()
conn.close()
###############################################################
for j in range(1):
  for i in range(0,len(ids)):   

    # Create your dictionary here (this is just an example)
    
      start_time = time.time()
      
      prof_id = ids[i]
      print(i,":",prof_id)
      res = web_scraper.scraper(prof_id=prof_id,llm = llm, embedding=embedding, key=key,ALL = ALL,verbose = verbose)
      
      end_time = time.time()
      duration = end_time - start_time

      # Open the file in append mode and write the dictionary
      res.dic["duration"] = duration

      with open('output_html.json', 'a') as file:
          json.dump(res.dic, file)
          file.write('\n')  # Add a newline to separate JSON objects




