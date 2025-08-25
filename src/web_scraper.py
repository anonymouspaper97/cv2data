import mysql.connector
import io
import info_extraction_2
import dbtools
from dbtools import connect_to_database
import asyncio


def web_extractor(prof_id):

    conn = conn = connect_to_database()
    cursor = conn.cursor()

    query = f"SELECT prof_id,extracted_cv,id, url FROM webpage_content WHERE prof_id = {prof_id} AND extracted_cv != '' and extracted_cv IS NOT NULL"

    cursor.execute(query)

    # Fetch the results
    rows = cursor.fetchall()
    
        
    text = ""
    urls = ""
    for row in rows:

        text += row[1]
        urls=urls+row[3]+" "
        
    urls.strip()         
    cursor.close()
    conn.close()
    return text,urls



def scraper(prof_id, llm, embedding,key, ALL = False,verbose = False):

    first_name,full_name = dbtools.personal_data(prof_id)

    content,url = web_extractor(prof_id)

    ext = info_extraction_2.extractor(content = content,first_name = first_name,full_name = full_name, llm = llm, key = key, ALL=ALL,document_type = 'html',verbose=verbose)

    
    ext.split(chunk_size=64000,chunk_overlap=200,length_function=len)
    if not ALL:
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

    temp = {"URL": f"{url}"}
    temp.update(ext.dic)
    ext.dic = temp


    # Applying the function to each category
    

    return ext 




