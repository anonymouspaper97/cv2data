import mysql.connector
import pdfplumber
import io
import info_extraction_2
import dbtools
from utils.dbtools import connect_to_database
import asyncio
from pypdf import PdfReader
from tika import parser

def pdf_extractor(prof_id):

    conn = conn = connect_to_database()
    cursor = conn.cursor()

    
    query = f"SELECT prof_id,raw_file,id FROM raw_files WHERE is_parsed = 0 AND prof_id = {prof_id} AND file_content = 'cv'"

    cursor.execute(query)

    # Fetch the results
    rows = cursor.fetchall()

    # Check if the result is an empty list
    if not rows:
        s = "No rows were found with the specified conditions."
        # Close the cursor and connection
        cursor.close()
        conn.close()
        print(s)
        return 
    
    else:
        
        text = ""
        
        for row in rows:

            try:
            
                pdf_data = row[1]
                pdf_stream = io.BytesIO(pdf_data)

                raw = parser.from_buffer(pdf_stream)
                text = text + raw['content'].strip() + "\n"

            
            except:

                print(f"error in parsing prof_id:{row[0]} and document_id:{row[2]}")
                
        # Close the cursor and connection
        cursor.close()
        conn.close()
        return text



def scraper(prof_id, llm, embedding,key, ALL = False):

    first_name,full_name = dbtools.personal_data(prof_id)

    content = pdf_extractor(prof_id)

    ext = info_extraction_2.extractor(content = content,first_name = first_name,full_name = full_name, llm = llm, key = key, ALL=ALL)

    
    ext.split(chunk_size=16000,chunk_overlap=200,length_function=len)
    if not ALL :
        ext.db(embedding=embedding)
    ext.gender_guesser()
    ext.lang_detector()
    ext.translate()
    ext.relevent_docs()

    async def extract():
        await ext.extract()
        # return results


    # Run the async function in the existing loop
    result = asyncio.run(extract())
   

    
    temp = {"prof_id": prof_id,"Full_Name": full_name}
    temp.update(ext.dic)
    ext.dic = temp 

    temp = {"URL": "PDF"}
    temp.update(ext.dic)
    ext.dic = temp


    # Applying the function to each category
    

    return ext 





