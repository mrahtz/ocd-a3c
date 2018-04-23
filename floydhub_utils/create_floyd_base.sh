#!/bin/bash

# Create a FloydHub dataset consisting of files changed by the setup wrapper
# (e.g. packages dependencies) for quicker launching of jobs

set -o errexit

touch before_file

dummy_cmd="true"
bash floydhub_utils/floyd_wrapper.sh $dummy_cmd

echo "Copying changed files..."
# '-type l': we need to copy the symbolic links set up by pipenv
find / \( -type l -o -type f \) -a -newer before_file | grep -v -e '^/proc' -e '^/sys' -e '^/output' -e '^/code' -e '^/floydlocaldata' | xargs -i cp --parents --no-dereference {} /output
echo "Done!"
