#!/bin/bash

# config the ssh keys for passwordless login beforehand
# change the path as needed

ssh $(whoami)@cygni-gw "rm -rf /users/micas/$(whoami)/no_backup/HeMAiAv2"
scp -r /users/micas/$(whoami)/no_backup/HeMAiAv2 $(whoami)@cygni-gw:/users/micas/$(whoami)/no_backup/
