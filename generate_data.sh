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
echo "done. now submitting jobs..."

#echo "test : "
#eai job submit \
#  --image registry.console.elementai.com/$ACCOUNT_ID/clutrr \
#  --data $ORG_NAME.$ACCOUNT_NAME.clutrr_data1:/data \
#  --data $ORG_NAME.$ACCOUNT_NAME.clutrr_code:/clutrr \
#  --cpu 2 \
#  --mem 8 \
#  -- bash -c "cd /clutrr/clutrr && python main.py --train_tasks 7.2,7.3,7.4 --test_tasks 7.2,7.3,7.4 --equal --data_name 'test' > out.txt 2>&1"

# while true; do sleep 60; done;
# cd /clutrr/clutrr && python main.py --train_tasks 7.2,7.3,7.4 --test_tasks 7.2,7.3,7.4 --equal --data_name 'test' > out.txt 2>&1

#################
# BIIIIIIG DATA #
#################

#echo "r3210-all+mem_l234 : "
#eai job submit \
#  --image registry.console.elementai.com/$ACCOUNT_ID/clutrr \
#  --data $ORG_NAME.$ACCOUNT_NAME.clutrr_data1:/data \
#  --data $ORG_NAME.$ACCOUNT_NAME.clutrr_code:/clutrr \
#  --cpu 8 \
#  --mem 90 \
#  --name "r3210-all+mem_l234" \
#  -- bash -c "cd /clutrr/clutrr && python main.py --train_tasks 7.2,7.3,7.4 --test_tasks 7.2,7.3,7.4,7.5,7.6,7.7,7.8,7.9,7.10 --train_rows 100000 --test_rows 10000 --equal --data_name 'r3210-all+mem_l234' > r3210-all+mem_l234.out 2>&1"
#echo "r3210-all+mem_l234_templatesplit : "
#eai job submit \
#  --image registry.console.elementai.com/$ACCOUNT_ID/clutrr \
#  --data $ORG_NAME.$ACCOUNT_NAME.clutrr_data1:/data \
#  --data $ORG_NAME.$ACCOUNT_NAME.clutrr_code:/clutrr \
#  --cpu 8 \
#  --mem 90 \
#  --name "r3210-all+mem_l234_templatesplit" \
#  -- bash -c "cd /clutrr/clutrr && python main.py --train_tasks 7.2,7.3,7.4 --test_tasks 7.2,7.3,7.4,7.5,7.6,7.7,7.8,7.9,7.10 --train_rows 100000 --test_rows 10000 --equal --template_split --data_name 'r3210-all+mem_l234_templatesplit' > r3210-all+mem_l234_templatesplit.out 2>&1"

#echo "r3210-all_l234 : "
#eai job submit \
#  --image registry.console.elementai.com/$ACCOUNT_ID/clutrr \
#  --data $ORG_NAME.$ACCOUNT_NAME.clutrr_data1:/data \
#  --data $ORG_NAME.$ACCOUNT_NAME.clutrr_code:/clutrr \
#  --cpu 8 \
#  --mem 90 \
#  --name "r3210-all_l234" \
#  -- bash -c "cd /clutrr/clutrr && python main.py --train_tasks 5.2,5.3,5.4 --test_tasks 5.2,5.3,5.4,5.5,5.6,5.7,5.8,5.9,5.10 --train_rows 100000 --test_rows 10000 --equal --data_name 'r3210-all_l234' > r3210-all_l234.out 2>&1"
#echo "r3210-all_l234_templatesplit : "
#eai job submit \
#  --image registry.console.elementai.com/$ACCOUNT_ID/clutrr \
#  --data $ORG_NAME.$ACCOUNT_NAME.clutrr_data1:/data \
#  --data $ORG_NAME.$ACCOUNT_NAME.clutrr_code:/clutrr \
#  --cpu 6 \
#  --mem 90 \
#  --name "r3210-all_l234_templatesplit" \
#  -- bash -c "cd /clutrr/clutrr && python main.py --train_tasks 5.2,5.3,5.4 --test_tasks 5.2,5.3,5.4,5.5,5.6,5.7,5.8,5.9,5.10 --train_rows 100000 --test_rows 10000 --equal --template_split --data_name 'r3210-all_l234_templatesplit' > r3210-all_l234_templatesplit.out 2>&1"


#echo "r0-facts_l234 : "
#eai job submit \
#  --image registry.console.elementai.com/$ACCOUNT_ID/clutrr \
#  --data $ORG_NAME.$ACCOUNT_NAME.clutrr_data1:/data \
#  --data $ORG_NAME.$ACCOUNT_NAME.clutrr_code:/clutrr \
#  --cpu 4 \
#  --mem 90 \
#  --name "r0-facts_l234" \
#  -- bash -c "cd /clutrr/clutrr && python main.py --train_tasks 1.2,1.3,1.4 --test_tasks 1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,1.10 --train_rows 100000 --test_rows 10000 --equal --data_name 'r0-facts_l234' > r0-facts_l234.out 2>&1"

echo "r0-facts_lALL_templatesplit_holdout : "
eai job submit \
  --image registry.console.elementai.com/$ACCOUNT_ID/clutrr \
  --data $ORG_NAME.$ACCOUNT_NAME.clutrr_data1:/data \
  --data $ORG_NAME.$ACCOUNT_NAME.clutrr_code:/clutrr \
  --cpu 4 \
  --mem 300 \
  --restartable \
  --name "r0-facts_lALL_templatesplit" \
  -- bash -c "cd /clutrr/clutrr && python main.py --train_tasks 1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,1.10 --test_tasks 1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,1.10 --train_rows 100000 --test_rows 10000 --equal --template_split --holdout --data_name 'r0-facts_lALL_templatesplit_holdout' > r0-facts_lALL_templatesplit_holdout.out 2>&1"

echo "r3210-all_lALL_templatesplit_holdout : "
eai job submit \
  --image registry.console.elementai.com/$ACCOUNT_ID/clutrr \
  --data $ORG_NAME.$ACCOUNT_NAME.clutrr_data1:/data \
  --data $ORG_NAME.$ACCOUNT_NAME.clutrr_code:/clutrr \
  --cpu 4 \
  --mem 300 \
  --restartable \
  --name "r3210-all_lALL_templatesplit" \
  -- bash -c "cd /clutrr/clutrr && python main.py --train_tasks 5.2,5.3,5.4,5.5,5.6,5.7,5.8,5.9,5.10 --test_tasks 5.2,5.3,5.4,5.5,5.6,5.7,5.8,5.9,5.10 --train_rows 100000 --test_rows 10000 --equal --template_split --holdout --data_name 'r3210-all_lALL_templatesplit_holdout' > r3210-all_lALL_templatesplit_holdout.out 2>&1"
