#!/usr/bin/python3

from atlassian import Jira
from pprint import pprint
from collections import OrderedDict 
import sys
import os
import re

JIRA_SERVER_HOME_CLOUD=0
JIRA_SERVER_HOME_LOCAL=1
JIRA_SERVER_AMD_CLOUD=2
JIRA_SERVER_AMD_LOCAL=3
JIRA_SERVER_AMD_ONTRACK=4
JIRA_SERVER_TO_USE=JIRA_SERVER_AMD_LOCAL

JIRA_SERVER_IDX_URL=0
JIRA_SERVER_IDX_USER=1
JIRA_SERVER_IDX_PASS=2
JIRA_SERVER_IDX_PROJECT=3

debug = 0

#   Converts to orderedDict type
#   input:
#   - dict - unordered dictionary.
#   return:
#       ordered dictionary
#       null - on any error.

def convertToOD(pDict):

    if type(pDict) != dict:
        print("ConvertToOD() Error: parameter is not dictionary type!.")
    sortedKeys = sorted(pDict)
    od = OrderedDict() 

    for i in sortedKeys:
        od[i] = pDict[i]
    return od

#   print dictionary recursively.

def printDictRecur(pDict, pIndent):
    debug = 0

    if debug:
        print("printDictRecur: entered: ", type(pDict))

    for i in pDict.items():
        if type(i[1]) == dict:

            if debug:
                print("dict encountered...", i[0])
    
            print(" --- ", pIndent, i[0], ": ")
            i1od=convertToOD(i[1])
            printDictRecur(i1od, pIndent + "  ")
        else:
            print(" --- ", pIndent, i)
        

jira_server={\
    JIRA_SERVER_HOME_CLOUD: ['',''],
    JIRA_SERVER_HOME_LOCAL: [\
    'http://10.0.0.100:8080', 'ggjira000', '8981555aaa'\
    'project = gg-proj-mgmt AND status IN ("To Do", "In Progress") ORDER BY issuekey'],

    JIRA_SERVER_AMD_CLOUD: [\
    'https://ggjira000.atlassian.net', 'ggjira100','8981555aaa', \
    'project = gg-proj-000 AND status IN ("To Do", "In Progress") ORDER BY issuekey'\
    ],
    JIRA_SERVER_AMD_LOCAL: [\
    'http://10.217.73.243:8080', 'ggjira200','8981555aaa', \
    'project = gg-proj-400 AND status IN ("To Do", "In Progress") ORDER BY issuekey'\
    ]
}

jira = Jira(
    url=jira_server[JIRA_SERVER_TO_USE][JIRA_SERVER_IDX_URL],
    username=jira_server[JIRA_SERVER_TO_USE][JIRA_SERVER_IDX_USER],
    password=jira_server[JIRA_SERVER_TO_USE][JIRA_SERVER_IDX_PASS])

JQL = jira_server[JIRA_SERVER_TO_USE][JIRA_SERVER_IDX_PROJECT]
data = jira.jql(JQL)

if type(data) == str and re.search("Basic auth with password is not allowed on this instance", data):
    print(data)
    quit(1)

issues=list(data.values())[-1]


if debug:
    for i in issues:
    	pprint(i)

print("==============================================")

issue1=issues[0]
indent = ""

#   convert to ordered dictionary (by keys).

issue1Od=convertToOD(issue1)

#   Recursively iterate over dictionary and print out.
#   The dictionary can be either
#   - key value pairs in which pair will be printed.
#   - key dictionary pair in which it recurs.

printDictRecur(issue1Od, indent)
