# This script shows how to use the client in anonymous mode
# against jira.atlassian.com.
#from jira import JIRA
import jira
import re
import os
import sys

debug=1
proj_qualifier_amd_jira="ontrack-internal.amd.com"
proj_qualified_ggjira000="https://ggjira000.atlassian.net"   
os.system("clear")

print(len(sys.argv))
if debug:
    for i in sys.argv:
        print(i)

jira_url="https://ggjira100.atlassian.net"
try:
	if sys.argv[1].strip() != "":
		jira_url=sys.argv[1].strip()
		print("Jira URL: ", jira_url)
except Exception as msg:
    print("No project specified in command line. Using default url: https://ggjira100.atlassian.net")
    print(msg)

jira_options={'server': jira_url + "/jira"}
jira = jira.JIRA('https://ggjira100.atlassian.net', basic_auth=("ggjira000", "61z2Rg0PX8INeZkasl5W75F4"))

print(jira)


print("===================")
print("obtaining projects...")

if jira_url == proj_qualifier_amd_jira:
    print("AMD jira...")
    project=jira.project('SWDEV')
elif jira_url == proj_qualified_ggjira000:
    project=jira.project('GP0')
else:
    projects=jira.projects()
print(projects)
print(len(projects))
#keys = sorted([project.key for project in projects])[2:5]

'''
print(projects)
print("obtaining keys...")
print(keys)
issue=jira.issue("issue-1000-1001")
print(issue)
'''

#issue=jira.issue("issue-1000-1001")
#print(issue)

#issues_in_proj = jira.search_issues('project=gg-proj-1001')

