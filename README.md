# cv2data
Extracts structured data from academic and professional CVs using LLMs. Leverages ChatGPT to parse unstructured documents (PDFs, text, web profiles) into standardized JSON for analyzing careers, positions, awards, and roles.

# Structured CV Extraction  

This repository contains code for extracting structured data from academic and professional CVs using large language models (LLMs). The pipeline leverages ChatGPT to parse unstructured documents (PDFs, text, and web profiles) into standardized JSON formats, enabling downstream analysis of career trajectories, academic positions, awards, and professional roles.  

## Features  
- Parse academic and professional CVs (PDF, text, HTML).  
- Extract structured information such as:  
  - Academic positions  
  - Professional roles  
  - Awards, fellowships, and honors  
  - Education history  
- Output in standardized JSON format.  
- Modular pipeline for easy extension and reproducibility.  

## Installation  
Clone the repository and install the dependencies:  

```bash
git clone https://github.com/anonymouspaper97/cv2data.git
cd cv2data
pip install -r requirements.txt
```
## Example output (simplified):

{
  "name": "Jane Doe",
  "positions": [
    {"title": "Assistant Professor", "institution": "University X", "years": "2020–present"}
  ],
  "awards": ["Best Dissertation Award 2019"],
  "education": ["PhD in Data Science, University Y, 2019"]
}


