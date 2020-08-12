#!/usr/bin/python3

# https://jira.readthedocs.io/en/master/examples.html#quickstart
from jira import JIRA
import re

JIRA_SERVER_IP={\
	"AMD_LOCAL": "http://10.217.73.243:8080", \
	"AMD_LOCAL_IXT39": "http://10.216.67.167:8080" \
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
jira = JIRA(basic_auth=(\
	JIRA_SERVER_AUTH_BASIC[JIRA_SERVER_PROFILE].split()[0].strip(), \
	JIRA_SERVER_AUTH_BASIC[JIRA_SERVER_PROFILE].split()[1].strip()))
#props = jira.application_properties()
#projects = jira.projects()

