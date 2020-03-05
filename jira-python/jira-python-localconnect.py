# This script shows how to use the client in anonymous mode
# against jira.atlassian.com.
#from jira import JIRA
import jira
import re
import os

os.system("clear")

jira_options={'server': "https://ggjira100.atlassian.net/jira}"}
jira = jira.JIRA('https://ggjira100.atlassian.net', basic_auth=("ggjira000", "61z2Rg0PX8INeZkasl5W75F4"))

#print(jira)

#print("===================")
#print("obtaining projects...")
#projects=jira.projects()

#for proj in projects:
#	print("proj: ", proj)


