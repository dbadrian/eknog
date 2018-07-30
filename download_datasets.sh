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


# PLEASE USE WITH CARE. Verify the downloaded datasets! There have been mistakes in the ConvE repository, which were manually fixed.
# We can not guarantee that the data is correct.

packagelist=(
    'https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:wordnet-mlj12.tar.gz'
    'https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:fb15k.tgz'
    'https://raw.githubusercontent.com/TimDettmers/ConvE/master/nations.tar.gz'
    'https://raw.githubusercontent.com/TimDettmers/ConvE/master/umls.tar.gz'
    'https://raw.githubusercontent.com/TimDettmers/ConvE/master/kinship.tar.gz'
    'https://raw.githubusercontent.com/TimDettmers/ConvE/master/WN18RR.tar.gz'
    'https://raw.githubusercontent.com/TimDettmers/ConvE/master/FB15k-237.tar.gz'
    'https://raw.githubusercontent.com/TimDettmers/ConvE/master/YAGO3-10.tar.gz'
)

rename_files () {
   mv $1/*test.txt $1/test.txt
   mv $1/*train.txt $1/train.txt
   mv $1/*valid.txt $1/valid.txt
}

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

for i in ${packagelist[*]}
do
  wget --directory-prefix=${SCRIPTPATH}/data/ --content-disposition --trust-server-names $i 2>&1  #$JEF is defined in my ~/.bashrc script
done

tar -zxf data/fb15k.tgz -C data && mv data/FB15k data/fb15k && rename_files data/fb15k
tar -zxf data/wordnet-mlj12.tar.gz -C data && mv data/wordnet-mlj12 data/wn18 && rename_files data/wn18
mkdir data/wn18rr && tar -zxf data/WN18RR.tar.gz -C data/wn18rr
mkdir data/yago3_10 && tar -zxf data/YAGO3-10.tar.gz -C data/yago3_10
mkdir data/fb15k_237 && tar -zxf data/FB15k-237.tar.gz -C data/fb15k_237
mkdir data/umls && tar -zxf data/umls.tar.gz -C data/umls
mkdir data/kinship && tar -zxf data/kinship.tar.gz -C data/kinship
mkdir data/nations && tar -zxf data/nations.tar.gz -C data/nations