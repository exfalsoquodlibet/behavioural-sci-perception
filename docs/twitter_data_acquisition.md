# Twitter data acquisition

## Source

We used the collection of tweets available from [TweetSets](https://tweetsets.library.gwu.edu/), an archive of Twitter datasets for research and archiving
managed by George Washing University.

The archived dataset we queried was their [Coronavirus dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/LW0BTB).

This dataset contains the tweet ids of 188,026,475 tweets related to Coronavirus or COVID-19. They were collected
between March 3, 2020 and May 1, 2020 from the Twitter API using Social Feed Manager. These tweets were collected using
the POST statuses/filter method of the Twitter Stream API, using the following tags: coronavirus, COVID-19,
epidemiology, pandemic.


## Filtering

Before extracting the tweet ids from TweetSets, we applied the following filter query so that only tweets whose text
contains either of these terms were selected:

nudge, nudging, nudge theory, david halpern, susan michie, richard amlot, thaler, sunstein, kahneman, behavioural science, behaviour change, behavior change, behavioral science, behavioural scientist, behavioral scientist, behavioural insight, behavioral insight, libertarian paternalism, choice architecture, choice architect, behavioural analysis, behavioral analysis, behavioral analyst, behavioural insights team, nudge unit, behavioural economics, behavioral economics, behavioural economist, behavioural policy, behavioral policy, irrational behaviour, irrational behavior

This resulted in a dataset of 14,962 tweets, corresponding to 0.008% of the initial dataset.


## Dataset key figures
 
Tweet counts: 	14,962 
First tweet: 	03-03-2020 
Last tweet: 	01-05-2020

Breakdown by type of tweets: 

- original tweets: 	1,487 (10%) 
- retweets: 		10,932 (73%) 
- quotes:			2,194 (15%) 
- replies: 			349 (2%)


Link to the filtered dataset:
[http://tweetsets.library.gwu.edu/dataset/56297bc8](http://tweetsets.library.gwu.edu/dataset/56297bc8)