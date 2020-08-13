#!/usr/bin/python3

# https://jira.readthedocs.io/en/master/examples.html#quickstart
#from jira import JIRA
from atlassian import Jira

CONFIG_JIRA_LIBRARY_JIRA=1
CONFIG_JIRA_LIBRARY_ATLASSIAN=2
CONFIG_JIRA_LIBRARY=CONFIG_JIRA_LIBRARY_ATLASSIAN

import re
CONFIG_ENABLE_NAT=1

if (CONFIG_ENABLE_NAT == 0):
	JIRA_SERVER_IP={\
		"AMD_LOCAL": "https://10.217.73.243:8080", \
		"AMD_LOCAL_IXT39": "http://10.216.67.167:8080" \
	}
else:
	JIRA_SERVER_IP={\
		"AMD_LOCAL": "https://10.217.73.243:8080", \
		"AMD_LOCAL_IXT39": "http://192.168.122.200:8080" \
	}

JIRA_SERVER_AUTH_BASIC={\
	"AMD_LOCAL": "ggjira200, 8981555aaa", \
	"AMD_LOCAL_IXT39": "ggjira300, 8981555aaa" \
	}

JIRA_SERVER_PROFILE="AMD_LOCAL_IXT39"
print("Loggin into: ", JIRA_SERVER_IP[JIRA_SERVER_PROFILE])
print("Logging with: ", \
	JIRA_SERVER_AUTH_BASIC[JIRA_SERVER_PROFILE].split()[0].strip(), "|", \
	JIRA_SERVER_AUTH_BASIC[JIRA_SERVER_PROFILE].split()[1].strip())
options = {"server": "https://" + JIRA_SERVER_IP[JIRA_SERVER_PROFILE]}


'''
jira = JIRA('https://' + JIRA_SERVER_IP[JIRA_SERVER_PROFILE], basic_auth=(\
        JIRA_SERVER_AUTH_BASIC[JIRA_SERVER_PROFILE].split()[0].strip(), \
        JIRA_SERVER_AUTH_BASIC[JIRA_SERVER_PROFILE].split()[1].strip()))
'''

if (CONFIG_JIRA_LIBRARY==CONFIG_JIRA_LIBRARY_ATLASSIAN):

    print("Using atlassian library...")

    jira = Jira(
        url='http://192.168.122.200:8080',
        username='ggjira300',
        password='8981555aaa')

    #props = jira.application_properties()
    #projects = jira.projects()
    
    JQL = 'project = gg-proj-000 AND status IN ("To Do", "In Progress") ORDER BY issuekey'
    data = jira.jql(JQL)
    
    # print(data)
    print(type(data))
    for i in range(0, len(data)):
            print("=")
            print(list(data.keys())[i])
            print("-")
            print(list(data.values())[i])
    
    '''
    issues=list(data.values())[-1]
    print(issues)
    
    for i in issues:
            pprint(i)
    
    '''
    
elif (CONFIG_JIRA_LIBRARY==CONFIG_JIRA_LIBRARY_JIRA):
    print("Using jira library")
    
    jira = JIRA(basic_auth=(\
	    JIRA_SERVER_AUTH_BASIC[JIRA_SERVER_PROFILE].split()[0].strip(), \
	    JIRA_SERVER_AUTH_BASIC[JIRA_SERVER_PROFILE].split()[1].strip()))
else:
    print("Unsupported library is enabled...")
    quit(1)

