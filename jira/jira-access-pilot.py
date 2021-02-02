#!/usr/bin/python3

# http://jira.readthedocs.io/en/master/examples.html#quickstart
from jira import JIRA
from atlassian import Jira
import pprint 
pp = pprint.PrettyPrinter(indent=4)

CONFIG_JIRA_LIBRARY_JIRA=1
CONFIG_JIRA_LIBRARY_ATLASSIAN=2
CONFIG_JIRA_LIBRARY=CONFIG_JIRA_LIBRARY_ATLASSIAN
CONFIG_JIRA_LIBRARY=CONFIG_JIRA_LIBRARY_JIRA

import re
CONFIG_ENABLE_NAT=1

if (CONFIG_ENABLE_NAT == 0):
	JIRA_SERVER_IP={\
		"AMD_LOCAL": "10.217.73.243:8080", \
		"AMD_LOCAL_IXT39": "10.216.67.167:8080", \
        "HOME_LOCAL": "11.0.0.29:8080", \
	    "AMD_CLOUD": "ggjira800.atlassian.net" \
	}
else:
	JIRA_SERVER_IP={\
		"AMD_LOCAL": "10.217.73.243:8080", \
		"AMD_LOCAL_IXT39": "192.168.122.200:8080", \
        "HOME_LOCAL": "11.0.0.29:8080", \
	    "AMD_CLOUD": "ggjira800.atlassian.net" \
	}

JIRA_SERVER_AUTHINFO={\
	"AMD_LOCAL": "ggjira200, 8981555aaa", \
	"AMD_LOCAL_IXT39": "guyen000@gmail.com, 55g2X3OiVPPtTtTydWkz0C45", \
	"HOME_LOCAL": "ggjira300, 8981555aaa", \
	"AMD_CLOUD": "ggjira100, vySlhhoBjoFav1Th2q9p4183" \
	}

JIRA_SERVER_PROFILE="AMD_CLOUD"
#JIRA_SERVER_PROFILE="AMD_LOCAL_IXT39"
print("Loggin into: ", JIRA_SERVER_IP[JIRA_SERVER_PROFILE])
print("Logging with: ", \
	JIRA_SERVER_AUTHINFO[JIRA_SERVER_PROFILE].split()[0].strip(), "|", \
	JIRA_SERVER_AUTHINFO[JIRA_SERVER_PROFILE].split()[1].strip())
options = {"server": "http://" + JIRA_SERVER_IP[JIRA_SERVER_PROFILE]}


if (CONFIG_JIRA_LIBRARY==CONFIG_JIRA_LIBRARY_ATLASSIAN):

    print("Using atlassian library...")

    jira = Jira(
        url="http://" + JIRA_SERVER_IP[JIRA_SERVER_PROFILE],
        username=JIRA_SERVER_AUTHINFO[JIRA_SERVER_PROFILE].split()[0].strip(),
        password=JIRA_SERVER_AUTHINFO[JIRA_SERVER_PROFILE].split()[1].strip())

    JQL = 'project = gg-proj-000'
    data = jira.jql(JQL)

    jql_request = 'project = gg-project-000 AND status NOT IN (Closed, Resolved) ORDER BY issuekey'
    issues = jira.jql(jql_request)
    pp.pprint(issues)
    print("issues data type: ")
    print(type(issues))

    projects=jira.projects(included_archived=None)

    #props = jira.application_properties()

elif (CONFIG_JIRA_LIBRARY==CONFIG_JIRA_LIBRARY_JIRA):
    print("Using jira library")
    
    jira = JIRA(basic_auth=(\
        JIRA_SERVER_AUTHINFO[JIRA_SERVER_PROFILE].split()[0].strip(), \
        JIRA_SERVER_AUTHINFO[JIRA_SERVER_PROFILE].split()[1].strip()), \
        options={'server':'http://' + JIRA_SERVER_IP[JIRA_SERVER_PROFILE]})

    projects = jira.projects()
    print(projects)

    '''
    jra = jira.project('GP0')
    print(jra.name)                 # 'JIRA'
    print(jra.lead.displayName)     # 'John Doe [ACME Inc.]'
    '''

else:
    print("Unsupported library is enabled...")
    quit(1)

