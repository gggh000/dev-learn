#!/usr/bin/python3

# https://jira.readthedocs.io/en/master/examples.html#quickstart
from jira import JIRA
import re

JIRA_SERVER_IP={"AMD_LOCAL": "http://10.217.73.243:8080"}
options = {"server": "https://" + JIRA_SERVER_IP["AMD_LOCAL"]}
jira = JIRA(basic_auth=("ggjira200", "8981555aaa"))
#props = jira.application_properties()
#projects = jira.projects()

