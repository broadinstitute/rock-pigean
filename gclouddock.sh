#!/usr/bin/env bash
project="nitrogenase-docker"
name="rock-pigean"
tag="1.0.0"
image="${name}:${tag}"
echo "Using Google project ${project}, Docker project ${name}, image tag ${image}"
echo "Cloud-building Docker image:"
full="gcr.io/${project}/${image}"
sudo docker rmi $full --force
gcloud builds submit --timeout=60m --project ${project} --tag $full
arg1=$1
if [ -n "$arg1" ];then
  sudo docker run -it $full "$arg1"
fi
echo "Done with $full $arg1"
