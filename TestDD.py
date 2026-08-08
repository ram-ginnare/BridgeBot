from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

web_context = search.run("Who is PM Modi?")

print(web_context)