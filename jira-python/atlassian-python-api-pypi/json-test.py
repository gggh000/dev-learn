#!/usr/bin/python3

from atlassian import Jira
from pprint import pprint
import sys
import os
import re

JIRA_SERVER_HOME_CLOUD=0
JIRA_SERVER_HOME_LOCAL=1
JIRA_SERVER_AMD_CLOUD=2
JIRA_SERVER_AMD_LOCAL=3
JIRA_SERVER_AMD_ONTRACK=4

JIRA_SERVER_IDX_URL=0
JIRA_SERVER_IDX_USER=1
JIRA_SERVER_IDX_PASS=2
JIRA_SERVER_IDX_PROJECT=3

jira_server={\
    JIRA_SERVER_HOME_CLOUD: ['',''],
    JIRA_SERVER_HOME_LOCAL: [\
    'http://10.0.0.100:8080', 'ggjira000', '8981555aaa'\
    'project = gg-proj-mgmt AND status IN ("To Do", "In Progress") ORDER BY issuekey'],

    JIRA_SERVER_AMD_CLOUD: [\
    'https://ggjira000.atlassian.net', 'ggjira100','8981555aaa', \
    'project = gg-proj-000 AND status IN ("To Do", "In Progress"\) ORDER BY issuekey'\
    ]
}

jira = Jira(
    url=jira_server[JIRA_SERVER_AMD_CLOUD][JIRA_SERVER_IDX_URL],
    username=jira_server[JIRA_SERVER_AMD_CLOUD][JIRA_SERVER_IDX_USER],
    password=jira_server[JIRA_SERVER_AMD_CLOUD][JIRA_SERVER_IDX_PASS])

'''
 jira = Jira(
-    url='http://10.0.0.100:8080',
-    username='ggjira000',
-    password='8981555aaa')
'''

JQL = jira_server[JIRA_SERVER_AMD_CLOUD][JIRA_SERVER_IDX_PROJECT]
#JQL = 'project = gg-proj-mgmt AND status IN ("To Do", "In Progress") ORDER BY issuekey'

data = jira.jql(JQL)

'''
# print(data)
print(type(data))
for i in range(0, len(data)):
	print("=============================")
	print(list(data.keys())[i])
	print("-----------------------------")
	print(list(data.values())[i])
'''
if re.search("Basic auth with password is not allowed on this instance", data):
    print(data)
    quit(1)

issues=list(data.values())[-1]
print(issues)

for i in issues:
	pprint(i)

issue1=issues[1]
for i in issue1.items():
	print("=============================")
	print("type: ", type(i), i)
	print("-----------------------------")
	try:
		print("inner type: ", type(i[1]))

		if type(i[1]) == dict:
			innerData=i[1]
			for j in innerData.items():
				print(" - ", j)
	except Exception as msg:
		print(msg)
		print("Idx error")

