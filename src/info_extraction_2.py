import tiktoken, os, re, langchain, openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from typing import Optional, List, Tuple, Dict, Any, TypedDict
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.vectorstores import Chroma
from langdetect import detect
import gender_guesser.detector as gender
from deep_translator import GoogleTranslator
from langchain.callbacks import get_openai_callback
from langchain.schema import Document
from examples import html_example,pdf_example,html_example_text
from langchain.docstore.document import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field  
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
import uuid

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
        self.first_name = first_name
        self.llm = llm        
        if key!=None:
            self.key = key
        
        if document_type=='pdf':

        
            self.example_text = pdf_example

            examples = [
                (
                    self.example_text,
                    Researcher3(phd=[("(University of Frankfurt (Main)","1966")],
                    professorship = [("apl. Professor","Humboldt Universität Berlin","2001")],
                    habilitation = [("Humboldt University Berlin","1995")])
                
                    
                ),
            ]
        
        elif document_type=='html':

            self.example_text = html_example_text

            examples = [
                (
                    self.example_text,
                    Researcher3(phd=None,
                    professorship = [("professor für rhythmologie m s sicherheit interventioneller verfahren w2","charité universitätsmedizin berlin", "2020")],
                    habilitation = [("charité universitätsmedizin berlin","2020")])
                
                    
                ),
            ]
            
        
        else:

            raise ValueError("document_type can be either pdf or html") 

        self.messages = []

        description="Extract information about a researcher called {self.full_name} from the extracted CV including the university that the researcher recieved their ALL PhD degree. And also the ALL professorship titles he/she hold plus the universities and institutes of those professorships. Note that, it should not be necessarily a full professorship. It can be any kind of professorship. And also the date when the researcher got those positions. Pay special attention to the difference between the 'Ruf' (offer) and the actual commencement dates of those positions.You have to look for the commencement dates. Also if they mentioned their habilitation period and the location of it. If you could not find a piece of information, always mention in in your output with 'unknown' with small letter 'u'. For dates, always output the full date IF IT IS AVAILABLE not just the year. And format it in this form: month/year. This is a task that requires high precision so carefully read the text and analyse the entire text before making decisions. I repeat, you have to read the ENTIRE text and do NOT stop as soon as you find answers for a given field.",


        for text, tool_call in examples:
                self.messages.extend(
                    tool_example_to_messages({"input": text, "tool_calls": [tool_call]})
                )

        prompt = ChatPromptTemplate.from_messages(
        [
                (
                    "system",
                    "You are an expert extraction algorithm. "
                    "Only extract relevant information from the text. "+description
                    
                ),
                # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
                MessagesPlaceholder("examples"),  # <-- EXAMPLES!
                # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
                ("human", "{text}"),
        ]
        )

        self.chain = prompt | llm.with_structured_output(
            schema=Researcher3,
            method="function_calling",
            include_raw=False
        )


    def split(self,chunk_size = 2000, chunk_overlap = 200, length_function = len):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap, length_function = length_function)
        self.docs = [Document(page_content=x) for x in text_splitter.split_text(self.text)]
        

    def db(self,embedding):

        if not self.all: 

            print("db started")
        
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
            
            handler = CustomHandler()

            self.document_extraction_results = self.chain.invoke({"text": self.matching_docs, "examples": self.messages}, config={"callbacks": [handler]})
            
            print(f"Total Tokens: {cb.total_tokens}")
            print(f"Prompt Tokens: {cb.prompt_tokens}")
            print(f"Completion Tokens: {cb.completion_tokens}")
            print(f"Successful Requests: {cb.successful_requests}")
            print(f"Total Cost (USD): ${cb.total_cost}")
            self.price = cb.total_cost
            self.tokens = cb.total_tokens

            self.dd = {i:[] for i in self.attr}

            for t in self.document_extraction_results:
                if getattr(self.document_extraction_results,t[0]):
                    self.dd[t[0]].extend(getattr(self.document_extraction_results,t[0]))

            self.dic = make_dic(self.dd,price=cb.total_cost,tokens=cb.total_tokens)
            

            for category in ['phd', 'positions', 'habilitation', 'professorship', 'full_professorship']:
                if category in self.dic:
                    self.dic[category] = remove_duplicates(self.dic[category])
            
            
            self.dic = remove_unknown_entries(self.dic)
            
            # format the database to be empty for the next persoon: 
            if self.all == False:
                self.vectordb._collection.delete(self.vectordb.get()['ids'])    
        
    def print_info(self):
         
        for key in list(self.dic.keys()):
                print(f"{key}: {self.dic[key]}")

    


class Researcher3(BaseModel):

    phd: Optional[List[tuple]] = Field(
        description=f"The University name and date where the researcher received their Ph.D. degree or degrees in case there are multiple PhDs. Follow this format: [(university_1, date_1), (university_2, date_2)]. When searching, remember: Ph.D. might be referred to as 'Promotion', 'Promotionsschrift', 'Doctoral', 'Doctorate' or 'Ph.D.', 'dissertation' or sometimes with 'Summa cum laude'. Only consider an explicitly mentioned Ph.D. degree, excluding, for instance, a master's thesis. The date should be when they completed all requirements for their Ph.D., including the defense of their dissertation. If the period is mentioned like this: 1899-1905, output only the latter, 1905. I REPEAT! Extract ALL the PhD degrees in case there are multiple degrees",
    )

    professorship: Optional[List[tuple]] = Field(
        description=f"Titles of ALL the professorship positions that the researcher held, the place of them and the date that the researcher officially began to work in those professorships. In a list of tuples form like this: [('professorship_title_1', 'place_1', 'date_1' ), ('professorship_title_2', 'place_2', 'date_2')]. Separate the title, place, and date into separate elements of a tuple as it is described. I emphasize that you have to look for all professorship titles in case there are multiple professorship titles. Note that a position like 'Attending', 'group leader', 'direktor' or anything similar or managerial positions are NOT I repeat, ARE NOT considered as professorship but a junior professorship or assistant professorship ARE acceptable for instance. So the word 'professor' or its German equivalent MUST be in the title and if not that position is NOT a professorship. Distinguish between administrative leadership roles and academic professorships, which are distinct in their nature. Pay special attention to the difference between the 'Ruf' (offer) and the actual commencement dates of those positions. Note that a position may have a 'Ruf' (offer) date and a commencement date. The 'Ruf' is when the offer for the position was given, and the commencement date is when the researcher officially began the professorship. Please take care to list the commencement dates, NOT the 'Ruf' dates in this task. If you want to report the 'Ruf', ALWAYS include the word 'Ruf' in your output."
    ) 
    habilitation: Optional[List[tuple]] = Field(
        description=f"The universities or institutes where the researcher did their habilitation and the periods of them in a list of tuples form like this: [(university 1, period 1), (university 2, period 2)]. Traditionally in Germany, a habilitation serves as a formal qualification needed to become a university professor. Along with the habilitation, one is awarded a teaching qualification (Lehrbefähigung) and (upon request) a teaching license (Lehrbefugnis). Be careful about this part and don't mistake it for other things.",
    )



class Example(TypedDict):
    """A representation of an example consisting of text input and expected tool calls.

    For extraction, the tool calls are represented as instances of pydantic model.
    """

    input: str  # This is the example text
    tool_calls: List[BaseModel]  # Instances of pydantic model that should be extracted


def tool_example_to_messages(example: Example) -> List[BaseMessage]:
    """Convert an example into a list of messages that can be fed into an LLM.

    This code is an adapter that converts our example to a list of messages
    that can be fed into a chat model.

    The list of messages per example corresponds to:

    1) HumanMessage: contains the content from which content should be extracted.
    2) AIMessage: contains the extracted information from the model
    3) ToolMessage: contains confirmation to the model that the model requested a tool correctly.

    The ToolMessage is required because some of the chat models are hyper-optimized for agents
    rather than for an extraction use case.
    """
    messages: List[BaseMessage] = [HumanMessage(content=example["input"])]
    openai_tool_calls = []
    for tool_call in example["tool_calls"]:
        openai_tool_calls.append(
            {
                "id": str(uuid.uuid4()),
                "type": "function",
                "function": {
                    # The name of the function right now corresponds
                    # to the name of the pydantic model
                    # This is implicit in the API right now,
                    # and will be improved over time.
                    "name": tool_call.__class__.__name__,
                    "arguments": tool_call.json(),
                },
            }
        )
    messages.append(
        AIMessage(content="", additional_kwargs={"tool_calls": openai_tool_calls})
    )
    tool_outputs = example.get("tool_outputs") or [
        "You have correctly called this tool."
    ] * len(openai_tool_calls)
    for output, tool_call in zip(tool_outputs, openai_tool_calls):
        messages.append(ToolMessage(content=output, tool_call_id=tool_call["id"]))
    return messages



class CustomHandler(BaseCallbackHandler):
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        self.formatted_prompts = "\n".join(prompts)
        #self.log.info(f"Prompt:\n{formatted_prompts}")



def get_country_of_university(key,university_name,position = None, name = None):
        
        if position==None and name==None:
            prompt = f"Where is {university_name} located? ONLY return the name of the country in your output. If it is a city, found the country where it is located. If you could not find the country write 'unknown' with the small letter 'u'."
        
        elif position!=None and name!=None:
            prompt = f"{name} hold/held a position called {position} at {university_name}. Where is {university_name} located? ONLY return the name of the country in your output. If it is a city, found the country where it is located. If you could not find the country write 'unknown' with small letter 'u'"


        try:
            chat = ChatOpenAI(temperature=0, openai_api_key=key, model_name="gpt-4-turbo-preview")
            sys_msg = SystemMessage(content="You are a helpful assistant that finds the country where universities and institutes are located.")
            user_msg = HumanMessage(content=prompt)
            result = chat.invoke([sys_msg, user_msg])
            return result.content
        except Exception as e:
            return str(e)



def make_dic(dic, tokens, price,first_name):

    for dic_key in list(dic.keys()):
        
        for index,item in enumerate(dic[dic_key]):
            if item[-2]!="unknown": #item[-2] is the position of the university name in the tuples
                dic[dic_key][index] +=(get_country_of_university(item[-2]),)
            else:
                dic[dic_key][index] +=("unknown",)

    for key in list(dic.keys()):
    
        if key=="phd" or key=="postdoc":
            for index,item in enumerate(dic[key]):
                ll = {}
                ll["Institute"] = item[0] 
                ll["Period"] = item[1]
                ll["Country"] = item[2] 
                dic[key][index] = ll
                
        elif key == "professorship":
            for index,item in enumerate(dic[key]):
                ll = {}
                ll["Position"] = item[0] 
                ll["Institute"] = item[1]
                ll["Period"] = item[2]
                ll["Country"] = item[3]
                dic[key][index] = ll

        elif key == "full_professorship":
            for index,item in enumerate(dic[key]):
                ll = {}
                ll["Position"] = item[0] 
                ll["Institute"] = item[1]
                ll["Period"] = item[2]
                ll["Country"] = item[3]
                dic[key][index] = ll
    
        elif key == "habilitation": 
            for index,item in enumerate(dic[key]):
                ll = {}
                ll["Institute"] = item[0] 
                ll["Period"] = item[1] 
                ll["Country"] = item[2] 
                dic[key][index] = ll
                        

    dic_2 = {}
            
    for key in dic.keys():
        temp={}
        for index,item in enumerate(dic[key]):


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
    
        dic_2[key] = temp

    d = gender.Detector()
    g = d.get_gender(first_name)
    dic_2["Gender"] = g


    dic_2["Total_Tokens"] = tokens
    dic_2["Total_Cost_(USD)"] = price


    for category in ['phd', 'positions', 'habilitation', 'professorship', 'full_professorship']:
        if category in dic_2:
            dic_2[category] = remove_duplicates(dic_2[category])
    
    dic_2 = remove_unknown_entries(dic_2)

    return dic_2