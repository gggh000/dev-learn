#!/usr/bin/python3

# https://jira.readthedocs.io/en/master/examples.html#quickstart
from jira import JIRA
from atlassian import Jira
import pprint 
pp = pprint.PrettyPrinter(indent=4)

CONFIG_JIRA_LIBRARY_JIRA=1
CONFIG_JIRA_LIBRARY_ATLASSIAN=2
CONFIG_JIRA_LIBRARY=CONFIG_JIRA_LIBRARY_ATLASSIAN

import re
CONFIG_ENABLE_NAT=1

if (CONFIG_ENABLE_NAT == 0):
	JIRA_SERVER_IP={\
		"AMD_LOCAL": "https://10.217.73.243:8080", \
		"AMD_LOCAL_IXT39": "http://10.216.67.167:8080", \
        "HOME_LOCAL": "http://11.0.0.29:8080" \
	}
else:
	JIRA_SERVER_IP={\
		"AMD_LOCAL": "https://10.217.73.243:8080", \
		"AMD_LOCAL_IXT39": "http://192.168.122.200:8080", \
        "HOME_LOCAL": "HTTP://11.0.0.29:8080" \
	}

JIRA_SERVER_AUTH_BASIC={\
	"AMD_LOCAL": "ggjira200, 8981555aaa", \
	"AMD_LOCAL_IXT39": "ggjira100, 8981555aaa", \
	"HOME_LOCAL": "ggjira300, 8981555aaa" \
	}

JIRA_SERVER_PROFILE="HOME_LOCAL"
print("Loggin into: ", JIRA_SERVER_IP[JIRA_SERVER_PROFILE])
print("Logging with: ", \
	JIRA_SERVER_AUTH_BASIC[JIRA_SERVER_PROFILE].split()[0].strip(), "|", \
	JIRA_SERVER_AUTH_BASIC[JIRA_SERVER_PROFILE].split()[1].strip())
options = {"server": "https://" + JIRA_SERVER_IP[JIRA_SERVER_PROFILE]}


if (CONFIG_JIRA_LIBRARY==CONFIG_JIRA_LIBRARY_ATLASSIAN):

    print("Using atlassian library...")

    jira = Jira(
        url=JIRA_SERVER_IP[JIRA_SERVER_PROFILE],
        username=JIRA_SERVER_AUTH_BASIC[JIRA_SERVER_PROFILE].split()[0].strip(),
        password=JIRA_SERVER_AUTH_BASIC[JIRA_SERVER_PROFILE].split()[1].strip())

    JQL = 'project = gg-proj-000'
    data = jira.jql(JQL)

    jql_request = 'project = gg-project-000 AND status NOT IN (Closed, Resolved) ORDER BY issuekey'
    issues = jira.jql(jql_request)
    pp.pprint(issues)
    print("issues data type: ")
    print(type(issues))

    projects=jira.projects(included_archived=None)

    #props = jira.application_properties()
    #projects = jira.projects()
    
elif (CONFIG_JIRA_LIBRARY==CONFIG_JIRA_LIBRARY_JIRA):
    print("Using jira library")
    
    '''
    jira = JIRA('https://' + JIRA_SERVER_IP[JIRA_SERVER_PROFILE], basic_auth=(\
        JIRA_SERVER_AUTH_BASIC[JIRA_SERVER_PROFILE].split()[0].strip(), \
        JIRA_SERVER_AUTH_BASIC[JIRA_SERVER_PROFILE].split()[1].strip()))
    '''

    jira = JIRA(basic_auth=(\
	    JIRA_SERVER_AUTH_BASIC[JIRA_SERVER_PROFILE].split()[0].strip(), \
	    JIRA_SERVER_AUTH_BASIC[JIRA_SERVER_PROFILE].split()[1].strip()))
else:
    print("Unsupported library is enabled...")
    quit(1)

