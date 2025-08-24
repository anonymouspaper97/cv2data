import tiktoken
import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
import os
from typing import List, Tuple
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langdetect import detect
import gender_guesser.detector as gender
from deep_translator import GoogleTranslator
from typing import List, Optional
from langchain.callbacks import get_openai_callback
from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text, Number
import pandas as pd
from pydantic import BaseModel, Field, validator
from kor import extract_from_documents, from_pydantic, create_extraction_chain
from langchain.schema import Document
import openai
import re 
from examples import html_example,pdf_example
from langchain.docstore.document import Document


langchain.debug = False
def normalize_string(s, keep_special_chars=False):
    """Normalize strings by converting to lower case, removing all non-alphanumeric characters (except spaces), and replacing multiple spaces with a single space."""
    if not keep_special_chars:
        s = re.sub(r"-", " ", s)
        s = re.sub(r'[^\w\s]', '', s)  # Remove non-alphanumeric characters except spaces
    s = re.sub(r'\s+', ' ', s).lower() # Replace multiple spaces with a single space and convert to lower case
    return s.strip()                   # Remove leading and trailing spaces

def remove_duplicates(category_data):
    unique_entries = {}
    for key, value in category_data.items():
        # Normalize the entry values for comparison, except the 'Period' field
        normalized_entry = {k: normalize_string(v, k == 'Period') for k, v in value.items() if isinstance(v, str)}
        identifier = tuple(normalized_entry.items())
        if identifier not in unique_entries:
            unique_entries[identifier] = key
    # Rebuild the category with only unique entries
    return {unique_entries[identifier]: dict(identifier) for identifier in unique_entries}

def token_num(text: str, llm = "gpt-4",verbose = False):
    encoding = tiktoken.encoding_for_model(llm)
    token_count = len(encoding.encode(text))
    if verbose:
        print(f"The text contains {token_count} tokens.")
    return (token_count)

def remove_unknown_entries(record):
    """
    Remove entries with 'unknown' in 'Period' or 'Institute' fields
    from 'phd', 'habilitation', and 'first_professorship' sections.
    """
    for section in ['phd', 'habilitation', 'professorship']:
        if section in record:
            # Iterate over a copy of the dictionary to allow modification during iteration
            for key, value in list(record[section].items()):
                if value.get('Period', '').lower() == 'unknown' or value.get('Position', '').lower() == 'unknown':
                    del record[section][key]

    return record

class extractor:

    def __init__(self, content, full_name,first_name ,llm, key = None, ALL = False, document_type = 'pdf',verbose = False):

        self.all = ALL 
        self.dic = {}
        self.text = content
        self.full_name = full_name
        self.academic_list = ["first_phd", "professorship","habilitation"]
        self.place_date = [("first_phd_university","phd_date"),("professorship_place", "professorship_date"),("habilitation", "habilitation_date")]
        self.attr = list(getattr(Researcher3,"__fields__").keys())
        #self.first_name = full_name.split(".")[-1].split()[0]
        self.first_name = first_name
        self.llm = llm
        if key!=None:
            self.key = key
        
        if document_type=='pdf':

        
            self.example_text = pdf_example

            self.schema,self.validator = from_pydantic(
            Researcher3,
            description="Extract information about a researcher with the given first name and last name from the extracted text of their personal web page including the university that the researcher recieved their FIRST PhD degree. And also the ALL professorship titles he/she hold plus the universities and institutes of those professorships. Note that, it should not be necessarily a full professorship. It can be any kind of professorship. And also the date when the researcher got those positions. Pay special attention to the difference between the 'Ruf' (offer) and the actual commencement dates of those positions.You have to look for the commencement dates. Also if they mentioned their habilitation period and the location of it. If you could not find a piece of information, always mention in in your output with 'unknown' with small letter 'u'. For dates, always output the full date IF IT IS AVAILABLE not just the year. And format it in this form: month/year. This is a task that requires high precision so carefully read the text and analyse the entire text before making decisions. I repeat, you have read the ENTIRE text and do NOT stop as soon as you find answers for a given field.",
            examples=[
                (
                    self.example_text,

                    {"phd": [("(University of Frankfurt (Main)","1966")],
                    "professorship": [("apl. Professor","Humboldt Universität Berlin","2001")], 
                    "habilitation":[("Humboldt University Berlin","1995")]
                    },
                    #"academic_positions_countries" : ["Germany", "Germany", "Germany", "Germany", "Germany"]},
                )
            ],
            many=False, 
            )
        
        elif document_type=='html':

            self.example_text = html_example

            self.schema,self.validator = from_pydantic(
            Researcher3,
            description="Extract information about a researcher with the given first name and last name from the extracted raw HTML of their personal web page including the university that the researcher recieved their FIRST PhD degree. And also the ALL professorship titles he/she hold plus the universities and institutes of those professorships. Note that, it should not be necessarily a full professorship. It can be any kind of professorship. And also the date when the researcher got those positions. Pay special attention to the difference between the 'Ruf' (offer) and the actual commencement dates of those positions.You have to look for the commencement dates. Also if they mentioned their habilitation period and the location of it. If you could not find a piece of information, always mention in in your output with 'unknown' with small letter 'u'. For dates, always output the full date IF IT IS AVAILABLE not just the year. And format it in this form: month/year. This is a task that requires high precision so carefully read the text and analyse the entire text before making decisions. I repeat, you have read the ENTIRE text and do NOT stop as soon as you find answers for a given field.",
            examples=[
                (
                    self.example_text,

                    {"phd": [("Institut für klinische Physiologie, Universität Tübingen", "unknown")],
                    "professorship": [("Professor für Rhythmologie m. S. Sicherheit interventioneller Verfahren (W2)", "Charité Universitätsmedizin Berlin", "2020")], 
                    "habilitation":[("Charité Universitätsmedizin Berlin", "2019")]
                    },
                    #"academic_positions_countries" : ["Germany", "Germany", "Germany", "Germany", "Germany"]},
                )
            ],
            many=False, 
            )
        
        else:

            raise ValueError("document_type can be either pdf or html") 



        self.chain = create_extraction_chain(
            llm,
            self.schema,
            encoder_or_encoder_class="json",
            input_formatter="triple_quotes",
            verbose=verbose
        )


    def split(self,chunk_size = 2000, chunk_overlap = 200, length_function = len):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap, length_function = length_function)
        self.docs = [Document(page_content=x) for x in text_splitter.split_text(self.text)]
        

    def db(self,embedding):

        if not self.all: 

            print("db started")
        
            #database_directory = './data'
            #for file_name in os.listdir(database_directory):
             #   file_path = os.path.join(database_directory, file_name)
              #  if os.path.isfile(file_path):
               #     os.remove(file_path)
        
            

            self.vectordb = Chroma.from_documents(self.docs, embedding = embedding)

            print("db ended")

            #self.vectordb.persist()
    
    def gender_guesser(self):
        d = gender.Detector()
        self.gender = d.get_gender(self.first_name)
    
    def lang_detector(self):
        self.language = detect(self.text)

    def translate(self):
        
        self.gt = GoogleTranslator(source='en', target=self.language)
        

    def relevent_docs(self):

        if not self.all: #this means the original documnet consists of information other than CV
            self.query = self.gt.translate(f"CV, resume, biography, or academic path of {self.full_name}. Include all the academic positions, the PhD phase, the postdoc phase, and the habilitation phase.")
            #self.matching_docs = self.vectordb.similarity_search(self.query,k=2)
            self.matching_docs = self.vectordb.search(search_type="mmr ", query= self.query, k=5)
            ss = set([item.page_content for item in self.matching_docs])

            l = []

            for item in ss:
                for i in self.matching_docs:
                    if i.page_content==item:
                        l.append(i)
                        break
        
            self.matching_docs = list(l)
        
        else:

            self.matching_docs = self.docs


    async def extract(self):

        
        #print(self.chain.prompt.format_prompt(text="[user input]").to_string())
        
        print("extraction started")
        with get_openai_callback() as cb:
            
            self.document_extraction_results = await extract_from_documents(
                self.chain, self.matching_docs, max_concurrency=5, use_uid=False, return_exceptions=True
            )

            #print(self.chain.prompt.format_prompt(text="[user input]").to_string())
            
            print(f"Total Tokens: {cb.total_tokens}")
            print(f"Prompt Tokens: {cb.prompt_tokens}")
            print(f"Completion Tokens: {cb.completion_tokens}")
            print(f"Successful Requests: {cb.successful_requests}")
            print(f"Total Cost (USD): ${cb.total_cost}")
            self.price = cb.total_cost
            self.tokens = cb.total_tokens

            


        
          
            
            

            #researcher = self.document_extraction_results[0].get('data', {}).get('researcher', [])

            self.dd = {i:[] for i in self.attr}

            #for r in self.document_extraction_results:
             #   for item in self.attr:
              #      data = r.get("data").get("researcher",{}).get(item)
               #     if data: 
                #        self.dd[item].extend(list(data))
                        

            #for item in self.place_date:
            
             #   l = len(self.dd[item[0]])
                
              #  remove_index = []
               # for k in range(l):
                #    n = 0
                 #   for i in item:
                  #      if self.dd[i][k]=="unknown":
                   #         n+=1
        
                    #if n==len(item):
                     #   remove_index.append(k)
    


                #for i in item:
                 #   self.dd[i] = [self.dd[i][k] for k in range(l) if k not in remove_index]
            
            
            for item in self.document_extraction_results:
                if "researcher3" in item["data"]:
                    for key in list(item["data"]["researcher3"].keys()):
                        self.dd[key].extend(item["data"]["researcher3"][key])
            
            for key in list(self.dd.keys()):
                
                listlist = []
                for x in self.dd[key]:
                    if type(x)==str:
                        listlist.append(tuple([x]))
                    else:
                        listlist.append(tuple(x))

                ll = sorted(set(listlist), key=listlist.index)

                ll_temp = list(ll)
                for item in ll:

                    
                    if len(item)==1:
                        
                        ll_temp.remove(item)
                        
                    else:
                        unk = 0
                        for j in item:
                            if j=="unknown":
                                unk+=1
                            else:
                                break

                        if unk==len(item):
                            
                            
                            ll_temp.remove(item)
                            
    
                self.dd[key] = ll_temp
            
            keys = list(self.dd.keys())
      
            self.dic = {key:self.dd[key] for key in keys}

           

            for key in list(self.dic.keys()):
                if key!="positions":
                    for index,item in enumerate(self.dic[key]):
                        
                        if item[-2]!="unknown": #item[-2] is the position of the university name in the tuples
                            self.dic[key][index] +=(self.get_country_of_university(item[-2]),)
                        else:
                            self.dic[key][index] +=("unknown",)
                elif key=="professorship":
                    for index,item in enumerate(self.dic[key]):
                        if item[-2]!="unknown": #item[-2] is the position of the university name in the tuples
                            self.dic[key][index] +=(self.get_country_of_university(item[-2],position=item[-3],name=self.full_name),)
                        else:
                            self.dic[key][index] +=("unknown",)
                
                elif key=="full_professorship":
                    for index,item in enumerate(self.dic[key]):
                        if item[-2]!="unknown": #item[-2] is the position of the university name in the tuples
                            self.dic[key][index] +=(self.get_country_of_university(item[-2],position=item[-3],name=self.full_name),)
                        else:
                            self.dic[key][index] +=("unknown",)
                    
            
            print(self.dic)
            for key in list(self.dic.keys()):
                if key=="phd" or key=="postdoc":
                    for index,item in enumerate(self.dic[key]):
                        ll = {}
                        ll["Institute"] = item[0] 
                        ll["Period"] = item[1]
                        ll["Country"] = item[2] 
                        self.dic[key][index] = ll
                
                elif key == "professorship":
                    for index,item in enumerate(self.dic[key]):
                        ll = {}
                        ll["Position"] = item[0] 
                        ll["Institute"] = item[1]
                        ll["Period"] = item[2]
                        ll["Country"] = item[3]
                        self.dic[key][index] = ll

                elif key == "full_professorship":
                    for index,item in enumerate(self.dic[key]):
                        ll = {}
                        ll["Position"] = item[0] 
                        ll["Institute"] = item[1]
                        ll["Period"] = item[2]
                        ll["Country"] = item[3]
                        self.dic[key][index] = ll
    
                elif key == "habilitation": 
                    for index,item in enumerate(self.dic[key]):
                        ll = {}
                        ll["Institute"] = item[0] 
                        ll["Period"] = item[1] 
                        ll["Country"] = item[2] 
                        self.dic[key][index] = ll
                        

            
            for key in self.dic.keys():
                temp={}
                for index,item in enumerate(self.dic[key]):
                    if key =="phd":
                        temp[f"phd_{index}"] = item
                    elif key=="postdoc":
                        temp[f"postdoc_{index}"] = item
                    elif key =="professorship":
                        temp[f"professorship_{index}"] = item
                    elif key =="full_professorship":
                        temp[f"full_professorship_{index}"] = item
                    elif key=="habilitation":
                        temp[f"habilitation_{index}"] = item
    
                self.dic[key] = temp

            
            self.dic["Gender"] = self.gender


            self.dic["Total_Tokens"] = self.tokens
            self.dic["Total_Cost_(USD)"] = self.price


            for category in ['phd', 'positions', 'habilitation', 'professorship', 'full_professorship']:
                if category in self.dic:
                    self.dic[category] = remove_duplicates(self.dic[category])
            
            
            
            #rint(self.dic)
            self.dic = remove_unknown_entries(self.dic)
            
            # format the database to be empty for the next persoon: 
            if self.all == False:
                self.vectordb._collection.delete(vectordb.get()['ids'])
            
            
            '''
            for item in self.academic_list:
                if self.dic.get(item) != None:
                    self.dic[item+"_countries"] = []
                    if isinstance(self.dic[item], str):
                        # If the input is a string, convert it to a list with a single element
                        self.dic[item] = [self.dic[item]]
                    for university_name in self.dic[item]:
                        self.dic[item+"_countries"].append(self.get_country_of_university(university_name))
            '''

            


  
    def get_country_of_university(self,university_name,position = None, name = None):
        openai.api_key = self.key
        
        if position==None and name==None:
            prompt = f"Where is {university_name} located? ONLY return the name of the country in your output. If it is a city, found the country where it is located. If you could not find the country write 'unknown' with the small letter 'u'."
        
        elif position!=None and name!=None:
            prompt = f"{name} hold/held a position called {position} at {university_name}. Where is {university_name} located? ONLY return the name of the country in your output. If it is a city, found the country where it is located. If you could not find the country write 'unknown' with small letter 'u'"

        #elif university_name=="unknown":
         #   prompt = f"{name} hold/held a position called {position}. Where is {university_name} located? ONLY return the name of the country in your output. If it is a city, found the country where it is located. If you could not find the country write 'unknown' "



        #response = openai.Completion.create(
        #engine="gpt-4",
        #prompt=prompt,
        #max_tokens=1024,
        #n=1,
        #stop=None,
        #temperature=0,
    #)

        try:
            response = openai.ChatCompletion.create(
            model="gpt-4",  # Replace with the specific GPT-4 model you have access to
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}]
                      
        )
            return response.choices[0].message.content

        except Exception as e:
            return str(e)
        
        
        #country = response.choices[0].text.strip()
        #return country



        
    def print_info(self):
         
        for key in list(self.dic.keys()):
                print(f"{key}: {self.dic[key]}")

    
class Researcher(BaseModel):

 
    #bachelor_university: Optional[str] = Field(
     #   description=f"The name of the university that the researcher got his/her bachelor's degree or any similar degree",
    #)

    #bachelor_country: Optional[str] = Field(
     #   description=f"The country of the bachelor's university",
    #)

    #master_university: Optional[str] = Field(
     #   description=f"The name of the university from which the researcher got his/her master's degree or anything similar",
    #)

    #master_country: Optional[str] = Field(
     #   description=f"The country of the master's university",
    #)

    phd_university: Optional[List[str]] = Field(
        description=f"The name of the university(s) that the researcher got his/her PhD degree in a list form",
    )

    phd_date: Optional[List[str]] = Field(
        description=f"The date or period that the researcher got their PhD in a list form",
    ) 

    #phd_country: Optional[str] = Field(
     #   description=f"The country that the PhD university",
    #)

    postdoc_universities: Optional[List[str]] = Field(
        description=f"The name of the university or universities that the researcher did post doc(s) in chronological order in a list form",
    )

    postdoc_date: Optional[List[str]] = Field(
        description=f"The period(s) of the postdoc(s) in chronological order in a list form",
    )

    #postdoc_countries: Optional[List[str]] = Field(
     #   description=f"The name of the countries of postdoc_universities a list form",
    #) 

    academic_positions: Optional[List[str]] = Field(
        description=f"Titles of all the academic positions that the researcher held",
    ) 

    academic_positions_places: Optional[List[str]] = Field(
        description=f"Places in which the researcher held academic titles",
    )

    academic_positions_date: Optional[List[str]] = Field(
        description=f"The period(s) of the academic position(s) in chronological order in a list form",
    )

    habilitation: Optional[List[str]] = Field(
        description=f"The univerisities or institutes that the researcher did their habilitation",
    )

    habilitation_date: Optional[List[str]] = Field(
        description=f"The period of the habilitation period",)



    #academic_positions_countries: Optional[List[str]] = Field(
     #   description=f"countries of the places in which the researcher held an academic position",
    #) 



    

    #@validator("first_name")
    #def name_must_not_be_empty(cls, v):
     #   if not v:
      #      raise ValueError("Name must not be empty")
       # return v


class Researcher2(BaseModel):

    phd: Optional[List[str]] = Field(
        description=f"The name of the university(s) and the period(s) that the researcher got his/her PhD degree in a list of tuples form like this: [(university 1, period 1), (university 2, period 2)]. Only include the data that is explicitly written as PhD or something similar.",
    )

    postdoc: Optional[List[str]] = Field(
        description=f"The name of the university(s) and the period(s) that the researcher did post doc(s) in a list of tuples form like this: [(university 1, period 1), (university 2, period 2)]. Only include the data that is explicitly written as post doc or something similar.",
    )

    positions: Optional[List[str]] = Field(
        description=f"Titles of all the positions that the researcher held, the places of those positions and the periods of those positions in a list of tuples form like this: [(position title 1, place 1, period 1), (position title 2, place 2, period 2)]. Do NOT mix up positions with educational degrees like Bachelor's, Master's or PhD. And place means the institude, university or organization which the researcher held that position.",
    ) 

    habilitation: Optional[List[str]] = Field(
        description=f"The univerisities or institutes that the researcher did their habilitation and the periods of them in a list of tuples form like this: [(university 1, period 1), (university 2, period 2)]. Traditionally in Germany, a habilitation serves as a formal qualification needed to become a university professor. Along with the habilitation, one is awarded teaching qualification (Lehrbefähigung) and (upon request) a teaching license (Lehrbefugnis). Be careful about this part and don't mistaken it with other things.",
    )


    
ex = {"phd_university": ["University of Magdeburg"],
                 "phd_date": ["2006"],
                 #"phd_country": "Germany",
                 "academic_positions": ["Wissenschaftlicher Mitarbeiter",
                                        "Vertretungsprofessur",
                                        "Lehrstuhlvertretung",
                                        "Lehrstuhlvertretung",
                                        "Inhaber des Lehrstuhls für Erziehungswissenschaft mit dem Schwerpunkt Berufspädagogik"],
                 "academic_positions_date": ["2000-2008","2008","2009-2011","2011-2012","from 2012"],
                 "academic_positions_places" : ["Universitäten Magdeburg und Darmstadt", 
                                                "Universität Kassel",
                                                "TU Darmstadt",
                                                "RWTH Aachen University",
                                                "RWTH Aachen University"],
                  "habilitation":["Otto-von-Guericke-Universität Magdeburg"],
                  "habilitation_date": ["2014"]}


ex2 = {"phd": [("University of Magdeburg","2006")],
                 "academic_positions": [("Wissenschaftlicher Mitarbeiter","Universitäten Magdeburg und Darmstadt","2000-2008"),
                                        ("Vertretungsprofessur","Universität Kassel","2008"),
                                        ("Lehrstuhlvertretung","TU Darmstadt","2009-2011"),
                                        ("Lehrstuhlvertretung","RWTH Aachen University","2011-2012"),
                                        ("Inhaber des Lehrstuhls für Erziehungswissenschaft mit dem Schwerpunkt Berufspädagogik","RWTH Aachen University","from 2012")],
                  "habilitation":[("Otto-von-Guericke-Universität Magdeburg","2014")]
                  }


class Researcher3(BaseModel):

    phd: Optional[List[tuple]] = Field(
        description=f"The University name and date where the researcher received their FIRST PhD degree. Follow this format: [(university, date)]. When searching, remember: PhD might be referred to as 'Promotion', 'Promotionsschrift', 'Doctoral', 'Doctorate' or 'PhD', or sometimes with 'Summa cum laude'. Only consider an explicitly mentioned PhD degree, excluding, for instance, a medical degree or MD thesis. The date should be when they completed all requirements for their PhD, including the defense of their dissertation. If the period is mentioned like this: 1899-1905, output only the latter, 1905.",
    )

    professorship: Optional[List[tuple]] = Field(
        description=f"Titles of the ALL the professorship positions that the researcher held, the place of them and the date that the researcher offically began to work in those professorships. In a list of tuples form like this: [('professorship_title_1', 'place_1' , 'date_1' ), ('professorship_title_2', 'place_2' , 'date_2')]. Separate the title, place, date into separate elements of a tuple as it is described. I emphasize that you have to look for the all professorships titles in case there are multiple professorship titles. Note that a position like 'Attending', 'group leader', 'direktor' or anything similar or managerial positions are NOT I repeat, ARE NOT considered as professorship but a junior professorship or assistant professorship ARE acceptable for instance. So the word 'professor' or its German equivalent MUST be in the title and if not that position is NOT a professorship. Distinguish between administrative leadership roles and academic professorships, which are distinct in their nature. Pay special attention to the difference between the 'Ruf' (offer) and the actual commencement dates of those positions. Note that a position may have a 'Ruf' (offer) date and a commencement date. The 'Ruf' is when the offer for the position was given, and the commencement date is when the researcher officially began the professorship. Please take care to list the commencement dates, NOT the 'Ruf' dates in this task. If you want to report the 'Ruf', ALWAYS include the word 'Ruf' in your output."
    ) 
    habilitation: Optional[List[tuple]] = Field(
        description=f"The univerisities or institutes that the researcher did their habilitation and the periods of them in a list of tuples form like this: [(university 1, period 1), (university 2, period 2)]. Traditionally in Germany, a habilitation serves as a formal qualification needed to become a university professor. Along with the habilitation, one is awarded teaching qualification (Lehrbefähigung) and (upon request) a teaching license (Lehrbefugnis). Be careful about this part and don't mistaken it with other things.",
    )


    
ex = {"phd_university": ["University of Magdeburg"],
                 "phd_date": ["2006"],
                  "habilitation":["Otto-von-Guericke-Universität Magdeburg"],
                  "habilitation_date": ["2014"]}


ex2 = {"phd": [("(University of Frankfurt (Main)","1966")],
        "first_professorship": [("apl. Professor","Humboldt Universität Berlin","2001")]         ,
                  "habilitation":[("Humboldt University Berlin","1995")]
                  }

#description=f"The name of the university and the period that date that researcher finished his/her FIRST PhD degree in a list of tuples form like this: [(university, date)]. Note that in Germany, most of the times PhD is called 'Promotion', 'Promotionsschrift', 'Doctoral', 'Doctorate','PhD' and similar words or sometimes with 'Summa cum laude'. So first look for these words. DO NOT consider a medical degree or any degree other than an explict PhD degree for this task. For the date, you MUST look for the date on which a PhD candidate successfully completes all the requirements for their doctoral degree, including the submission and defense of their dissertation. So, if there is a date when they got their degree and a date for when they defended their thesis, you have to output the second. Also if they menthined the date in the form of a period like this: 1899 - 1905, your output should be 1905. DO NOT infere the university based on the information of other parts of the CV. Only mention the university or institude of the PhD if it's directly mentioned in the same part of the text."
#Note that for extracting the date, you have to look for the date that the researcher began his/her job as a professor and not 'Ruf', which is the point at which a researcher is offered a professorship, but it is NOT the exact moment they officially become a professor and we are NOT looking for this date.