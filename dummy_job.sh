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

# --volume /mnt/datasets/public/nicolasg/clutrr:/clutrr \
# --data $ORG_NAME.$ACCOUNT_NAME.clutrr1_valid:/data/clutrr/data/data_r0-facts_lALL_templatesplit_holdout_1581431566.9509876/1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,1.10_valid \
# --data $ORG_NAME.$ACCOUNT_NAME.clutrr1_test:/data/clutrr/data/data_r0-facts_lALL_templatesplit_holdout_1581431566.9509876 \
