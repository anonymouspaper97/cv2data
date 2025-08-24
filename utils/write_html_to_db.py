from scraper import page,is_valid_webpage
import os
import pandas as pd
from dbtools import connect_to_database
import asyncio  
import validators
import requests
from urllib.parse import urlparse

import time

## Purpose of this Script
'''
This script retrieves webpage records marked as HTML from a database, scrapes their raw HTML 
content using asynchronous scraping tools, and updates the database with the extracted HTML.  
It automates the process of collecting and storing webpage source code for further analysis or processing.
'''

# Define the main async function
async def main(p):
    async def scrape_playwright():
        results = await p.ascrape()
        return results
    


    r = await scrape_playwright()
    
    return r
    

conn = connect_to_database()
cursor = conn.cursor()

sql_query = """
SELECT * FROM webpage_content WHERE is_html=1
"""

# Execute the query and store the result in a DataFrame
df_web = pd.read_sql_query(sql_query, conn)

conn.close()


# Run the main function using asyncio

total = len(df_web)

if __name__ == "__main__":
    start_time = time.time()

    conn = connect_to_database()
    cursor = conn.cursor()

    for index, row in df_web.iterrows():

        url = row.url

        print(f"{index}/{total}")
    
        result = asyncio.run(main(page(url)))

        update_query = "UPDATE webpage_content SET html_content = %s WHERE id = %s"
        
        cursor.execute(update_query, (result["html_raw"], row.id))

        print(f"{index}/{total}")
    
    # Commit the updates to the database
    conn.commit()

    # Close the cursor and the connection
    cursor.close()
    conn.close()

    end_time = time.time()

    # Calculate the difference
    execution_time = end_time - start_time

    print(f"The execution time is {execution_time} seconds")

       