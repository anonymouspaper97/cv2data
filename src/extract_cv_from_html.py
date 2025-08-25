## Purpose of this Script
'''
This script processes researcher webpages stored in a database, applies text-cleaning and chunking, and uses OpenAI language models (via LangChain) to extract structured curriculum vitae (CV) information.  
The extracted data is written back into the database, along with usage cost and execution time for each processed record.  
Its purpose is to automate CV data extraction from unstructured personal webpages into a standardized, machine-readable format for downstream analysis.
'''

from langchain.callbacks.base import BaseCallbackHandler
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any
import os 
from utils.dbtools import connect_to_database
import pandas as pd 
import time 
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.text_splitter import TokenTextSplitter
import re
import warnings
from langchain.schema.document import Document
import langchain 
#from langchain.chains import LLMChain
langchain.debug = False

# Filter out UserWarning raised by pandas
warnings.filterwarnings("ignore", category=UserWarning)

def split(text,chunk_size = 2000, chunk_overlap = 200, length_function = len):
        text_splitter = TokenTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
        docs = [Document(page_content=x) for x in text_splitter.split_text(text)]
        return docs


class CustomHandler(BaseCallbackHandler):
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        self.formatted_prompts = "\n".join(prompts)
        print(f"Prompt:\n{self.formatted_prompts}")
        #self.log.info(f"Prompt:\n{formatted_prompts}")


def main():
    conn = connect_to_database()
    sql_query = f"SELECT id,prof_id,plain_text_style_removed,url FROM webpage_content WHERE plain_text_style_removed IS NOT NULL"
    df_1 = pd.read_sql_query(sql_query, conn)
    conn.close()

    key = "YOUR KEY"

    os.environ["OPENAI_API_KEY"] = key

    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0,
    )

    embedding=OpenAIEmbeddings()



    handler = CustomHandler()

    conn = connect_to_database()
    cursor = conn.cursor()

    total = len(df_1)

    for index,row in df_1.iloc[0:].iterrows():
        
        #print(row.id,row.prof_id,row.plain_text)

        sql_query = f"SELECT title, only_name FROM professor WHERE id = {row.prof_id}"


        df_2 = pd.read_sql_query(sql_query, conn)

        full_name = df_2.loc[0].title + df_2.loc[0].only_name

        prompt_CV = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert extraction algorithm. "
                "Only extract relevant information from the text. "
                f"Output all the information about {full_name} based on the text provided."
                "Ignore the list of publications."
                "The text is extracted from the HTML content of the personal webpage of the researcher so it also consists of not relevant information so ignore them."
                "Do not alter the text just give it structure."
                "If you could not find any information about the person, return NULL"
                
            ),
            # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
            #MessagesPlaceholder("examples"),  # <-- EXAMPLES!
            # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
            ("human", "{text}"),
        ]
    )


        runnable_CV = prompt_CV | llm
        content = row.plain_text_style_removed
        content = re.sub(r'(\n\s*){3,}', '\n\n', content)
        content = re.sub(r'[\xa0 ]{3,}', '\n\n', content)


        texts = split(content,chunk_size=128000)

        cv_content = ""
        total_cost = 0
        total_time = 0

        for text in texts:

            with get_openai_callback() as cb:

                start_time = time.time()


                cv = runnable_CV.invoke({"text": text, }, config={"callbacks": [handler]})

                end_time = time.time()

                execution_time = end_time - start_time

                if cv.content != "NULL":
                    cv_content = cv_content + cv.content + "\n"
                
                total_cost += cb.total_cost
                total_time +=execution_time

        update_query = "UPDATE webpage_content SET extracted_cv = %s, cv_price = %s, cv_duration = %s WHERE id = %s"
        cursor.execute(
            update_query,
            (cv_content.strip() if cv_content else None, float(total_cost), float(total_time), int(row.id))
            )
            
        conn.commit()
        print(f"\r{index}/{total}", end = "")
        
    cursor.close()
    conn.close()

if __name__ == "__main__":

    main()
