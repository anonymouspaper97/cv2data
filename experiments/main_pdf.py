from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import openai
import pandas as pd 
from langchain.llms import OpenAI
import os 

from utils.dbtools import connect_to_database
import json 

import utils.pdf_scraper 
import time 

prof_id = 1 

key = "YOUR KEY"
os.environ["OPENAI_API_KEY"] = key


llm = ChatOpenAI(
     model_name="gpt-4-turbo-preview",
    temperature=0,
)

embedding=OpenAIEmbeddings() 



ALL = True # for PDFs we assume, all the information is relevant

###############################################################

conn = conn = connect_to_database()
cursor = conn.cursor()

query = f"SELECT prof_id FROM raw_files WHERE file_content = 'cv'"

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
    
      start_time = time.time()
      
      prof_id = ids[i]
      print(i,":",prof_id)
      res = pdf_scraper.scraper(prof_id=prof_id,llm = llm, embedding=embedding, key=key,ALL = ALL)
      
      end_time = time.time()
      duration = end_time - start_time

      # Open the file in append mode and write the dictionary
      res.dic["duration"] = duration
      with open('output_4.json', 'a') as file:
          json.dump(res.dic, file)
          file.write('\n')  # Add a newline to separate JSON objects




