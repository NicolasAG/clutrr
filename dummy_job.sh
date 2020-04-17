#!/usr/bin/env bash

# get organization / user / account name
export ORG_NAME=$(eai organization get --fields name --no-header)
export USER_NAME=$(eai user get --fields name --no-header)
export ACCOUNT_NAME=$(eai account get --fields name --no-header)
#export ACCOUNT_ID=$(eai account get --fields id --no-header)
export ACCOUNT_ID=$ORG_NAME.$ACCOUNT_NAME

echo "account id: $ACCOUNT_ID"

echo "pushing all files to code_clutrr ..."
all_files=$(ls -I data -I .git -I . -I ..)  # ignore data, .git, current and parent folders
for f in $all_files
do
  eai data push "code_clutrr" $f:$f
done
echo "done. now submitting job..."

try=1

eai job submit \
  --image registry.console.elementai.com/$ACCOUNT_ID/clutrr \
  --data $ORG_NAME.$ACCOUNT_NAME.data_clutrr1:/data \
  --data $ORG_NAME.$ACCOUNT_NAME.code_clutrr:/clutrr \
  --mem 12 \
  -- bash -c "while true; do sleep 60; done;"
  #--name "explore_csv_noproof_facts_proper_try${try}" \
  #-- bash -c "cd clutrr/ && python explore_csv.py"
  #-- bash -c "cd clutrr/ && python make_test_PROPER.py"

eai job exec --last -- bash
#eai job log -f --last
