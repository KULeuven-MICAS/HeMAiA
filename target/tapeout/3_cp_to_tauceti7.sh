#!/bin/bash

# config the ssh keys for passwordless login beforehand
# change the path as needed
scp -r /users/micas/$(whoami)/no_backup/HeMAiAv2 $(whoami)@cygni-gw:/users/micas/$(whoami)/no_backup/HeMAiAv2
