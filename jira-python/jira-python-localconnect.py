# This script shows how to use the client in anonymous mode
# against jira.atlassian.com.
#from jira import JIRA
import jira
import re
import os

os.system("clear")

jira_options={'server': "https://ggjira100.atlassian.net/jira}"}
jira = jira.JIRA('https://ggjira100.atlassian.net', basic_auth=("ggjira000", "61z2Rg0PX8INeZkasl5W75F4"))

print(jira)

print("===================")
print("obtaining projects...")
projects=jira.projects()
keys = sorted([project.key for project in projects])[2:5]

'''
print(projects)
print("obtaining keys...")
print(keys)
issue=jira.issue("issue-1000-1001")
print(issue)
'''

issue=jira.issue("issue-1000-1001")
#print(issue)

#issues_in_proj = jira.search_issues('project=gg-proj-1001')

