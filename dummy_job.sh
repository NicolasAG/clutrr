#!/usr/bin/env bash

# get organization / user / account name
export ORG_NAME=$(eai organization get --fields name --no-header)
export USER_NAME=$(eai user get --fields name --no-header)
export ACCOUNT_NAME=$(eai account get --fields name --no-header)
#export ACCOUNT_ID=$(eai account get --fields id --no-header)
export ACCOUNT_ID=$ORG_NAME.$ACCOUNT_NAME

echo "account id: $ACCOUNT_ID"

echo "pushing all files to data clutrr_code ..."
all_files=$(ls -I data -I .git -I . -I ..)  # ignore data, .git, current and parent folders
for f in $all_files
do
  eai data push "clutrr_code" $f:$f
done
echo "done. now submitting job..."

eai job submit \
  --image registry.console.elementai.com/$ACCOUNT_ID/clutrr \
  --data $ORG_NAME.$ACCOUNT_NAME.clutrr_data1:/data \
  --data $ORG_NAME.$ACCOUNT_NAME.clutrr_code:/clutrr \
  --mem 12 \
  -- bash -c "while true; do sleep 60; done;"
