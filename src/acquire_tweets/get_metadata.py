"""

This script will take the .txt file with the tweet ids, which must have one tweet id per line, and hydrate all those tweets.

Script adapted from: https://github.com/thepanacealab/SMMT/blob/master/data_acquisition/get_metadata.py
Developed during Biomedical Hackathon 6 - http://blah6.linkedannotation.org/
Authors: Ramya Tekumalla, Javad Asl, Juan M. Banda
Contributors: Kevin B. Cohen, Joanthan Lucero

How to run this script
----------------------
Arguments	Description	            Required
-i	        input text file name	A text file having one tweet id per line
-o	        output file name	    4 output files will be created using the given output file name

Usage : python -m src.acquire_tweets.get_metadata -i data/raw/tweet-ids-001.txt -o data/raw/hydrated_tweets

Output
------
You will get four output files:

1. a hydrated_tweets.json file which contains the full json object for each of the hydrated tweets
2. a hydrated_tweets.CSV file which contains partial fields extracted from the tweets.
3. a hydrated_tweets.zip file which contains a zipped version of the tweets_full.json file.
4. a hydrated_tweets_short.json which contains a shortened version of the hydrated tweets.

"""

import tweepy
import json
import math
import csv
import zipfile
import argparse
import os
import os.path as osp
import pandas as pd
from time import sleep

DIR_SECRETS = os.environ.get("DIR_SECRETS")
API_KEYS = os.path.join(DIR_SECRETS, 'api_keys.json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o",
                        "--outputfile",
                        help="Output file name with extension")
    parser.add_argument("-i",
                        "--inputfile",
                        help="Input file name with extension")
    parser.add_argument("-c",
                        "--idcolumn",
                        help="tweet id column in the input file, string name")

    args = parser.parse_args()
    if args.inputfile is None or args.outputfile is None:
        parser.error("please add necessary arguments")

    print(args.inputfile)

    with open(API_KEYS) as f:
        keys = json.load(f)

    auth = tweepy.OAuthHandler(keys['consumer_key'], keys['consumer_secret'])
    auth.set_access_token(keys['access_token'], keys['access_token_secret'])
    api = tweepy.API(auth)

    output_file = args.outputfile
    output_file_noformat = output_file.split(".", maxsplit=1)[0]
    print(output_file)
    output_file = '{}'.format(output_file)
    output_file_short = '{}_short.json'.format(output_file_noformat)
    compression = zipfile.ZIP_DEFLATED
    ids = []

    if '.txt' in args.inputfile:
        inputfile_data = pd.read_csv(args.inputfile, sep='\t')
        print('tab seperated file, using \\t delimiter')
    elif '.csv' in args.inputfile:
        inputfile_data = pd.read_csv(args.inputfile)

    ids = inputfile_data.squeeze().tolist()

    print('total ids: {}'.format(len(ids)))

    start = 0
    end = 100
    limit = len(ids)
    i = int(math.ceil(float(limit) / 100))

    last_tweet = None
    if osp.isfile(args.outputfile):
        with open(output_file, 'rb') as f:
            # may be a large file, seeking without iterating
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b'\n':
                f.seek(-2, os.SEEK_CUR)
            last_line = f.readline().decode()
        last_tweet = json.loads(last_line)
        start = ids.index(last_tweet['id'])
        end = start + 100
        i = int(math.ceil(float(limit - start) / 100))

    print('metadata collection complete')
    print('creating master json file')
    try:
        with open(output_file, 'a') as outfile:
            for go in range(i):
                print('currently getting {} - {}'.format(start, end))
                sleep(6)  # needed to prevent hitting API rate limit
                id_batch = ids[start:end]
                start += 100
                end += 100
                tweets = api.statuses_lookup(id_batch, tweet_mode='extended')
                for tweet in tweets:
                    json.dump(tweet._json, outfile)
                    outfile.write('\n')
    except Exception:
        print('exception: continuing to zip the file')

    print('creating ziped master json file')
    zf = zipfile.ZipFile('{}.zip'.format(output_file_noformat), mode='w')
    zf.write(output_file, compress_type=compression)
    zf.close()

    def is_retweet(entry):
        return 'retweeted_status' in entry.keys()

    def get_source(entry):
        if '<' in entry["source"]:
            return entry["source"].split('>')[1].split('<')[0]
        else:
            return entry["source"]

    print('creating minimized json master file')
    with open(output_file_short, 'w') as outfile:
        with open(output_file) as json_data:
            for tweet in json_data:
                data = json.loads(tweet)
                print(data.keys())  #
                t = {
                    "created_at": data["created_at"],
                    "text":
                    data["text"],  # TODO: could be "text" or "full_text"
                    "in_reply_to_screen_name": data["in_reply_to_screen_name"],
                    "retweet_count": data["retweet_count"],
                    "favorite_count": data["favorite_count"],
                    "source": get_source(data),
                    "id_str": data["id_str"],
                    "is_retweet": is_retweet(data),
                    "lang": data["lang"]
                }
                json.dump(t, outfile)
                outfile.write('\n')

    f = csv.writer(open('{}.csv'.format(output_file_noformat), 'w'))
    print('creating CSV version of minimized json master file')
    fields = [
        "favorite_count", "source", "text", "in_reply_to_screen_name",
        "is_retweet", "created_at", "retweet_count", "lang", "id_str"
    ]
    f.writerow(fields)
    with open(output_file_short) as master_file:
        for tweet in master_file:
            data = json.loads(tweet)
            f.writerow([
                data["favorite_count"], data["source"],
                data["text"].encode('utf-8'), data["in_reply_to_screen_name"],
                data["is_retweet"], data["created_at"], data["retweet_count"],
                data["lang"], data["id_str"].encode('utf-8')
            ])


# main invoked here
main()
