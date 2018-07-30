#!/bin/bash
# Copyright 2018 David B. Adrian, Mercateo AG (http://www.mercateo.com)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

prompt="Please select a log folder:"
options=( $(find logs -maxdepth 1 -print0 | xargs -0) )

PS3="$prompt "
select opt in "${options[@]}" "Quit" ; do
    if (( REPLY == 1 + ${#options[@]} )) ; then
        exit

    elif (( REPLY > 0 && REPLY <= ${#options[@]} )) ; then
        echo  "Killing existing tensorboards and starting screen on $opt"
        pkill tensorboard
        screen -d -m -S tensorboard bash -c 'cd $HOME/eknog && source activate tensorflow_p36 && tensorboard --logdir='${opt}'/tf_events --host=0.0.0.0 --port=8080'
        PUBLIC_DNS=($(ec2metadata --public-hostname))
        echo "URL: http://${PUBLIC_DNS}:8080"
        break

    else
        echo "Invalid option. Try another one."
    fi
done
