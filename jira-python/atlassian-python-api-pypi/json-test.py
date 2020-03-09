#!/usr/bin/python3

from atlassian import Jira
from pprint import pprint
import sys
import os

jira = Jira(
    url='http://10.0.0.100:8080',
    username='ggjira000',
    password='8981555aaa')
JQL = 'project = gg-proj-mgmt AND status IN ("To Do", "In Progress") ORDER BY issuekey'
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
				print(j)
	except Exception as msg:
		print(msg)
		print("Idx error")

