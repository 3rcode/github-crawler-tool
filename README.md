# Survey of some Github repositories

**This project serves for a survey of Github repositories about refering commit from a changelog**

## Structure

1. "base_functions.py" provides function to apply a function for all repositories in "data/Repos.csv" file
2. "make_data.py" provides functions to intialize data of repositories in "data/Repos.csv" file
3. "sample_data.py" provides functions to summarize and sample data 
4. "insight.py" provides some functions to describe data

## How to use?

1. Fill all repositories that you concern about in data/Repos.csv
2. Replace your Github token in settings.py
3. Clone it using function in make_data.py
4. Crawl commits, changelogs
5. You can now use some functions in this repo to derive infomation 

