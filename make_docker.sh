#!/usr/bin/env bash

: '
docker build -t clutrr .
docker tag clutrr images.borgy.elementai.net/nicolasg/clutrr
docker push images.borgy.elementai.net/nicolasg/clutrr
'

# get organization / user / account name
export ORG_NAME=$(eai organization get --fields name --no-header)
export USER_NAME=$(eai user get --fields name --no-header)
export ACCOUNT_NAME=$(eai account get --fields name --no-header)
#export ACCOUNT_ID=$(eai account get --fields id --no-header)
export ACCOUNT_ID=$ORG_NAME.$ACCOUNT_NAME

echo "account id: $ACCOUNT_ID"

export IMAGE=registry.console.elementai.com/$ACCOUNT_ID/clutrr

#echo "organization name: $ORG_NAME"
#echo "account name: $ACCOUNT_NAME"
#echo "account id: $ACCOUNT_ID"
#echo "image: $IMAGE"

docker build -t $IMAGE .
docker push $IMAGE