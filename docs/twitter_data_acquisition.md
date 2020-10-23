# Twitter data acquisition

## Source

We used the collection of tweets available from [TweetSets](https://tweetsets.library.gwu.edu/), an archive of Twitter datasets for research and archiving
managed by George Washing University.

The archived dataset we queried was their [Coronavirus dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/LW0BTB).

This dataset contains the tweet ids of 239,861,658 tweets related to Coronavirus or COVID-19. They were collected
between March 3, 2020 and June 9, 2020 from the Twitter API using Social Feed Manager. These tweets were collected using
the POST statuses/filter method of the Twitter Stream API, using the following tags: coronavirus, COVID-19,
epidemiology, pandemic.


## Filtering

Before extracting the tweet ids from TweetSets, we applied the following filter query so that only tweets whose text
contains either of these terms were selected:

nudge theory, david halpern, nick chater, susan michie, richard thaler, cass sunstein, dan kahneman, daniel kahneman, behavioural science, behaviour change, behavioural scientist,  behavioural insight, libertarian paternalism, choice architecture, choice architect, behavioural analysis, behavioural analyst, behavioural insights team, nudge unit, behavioural economics, behavioural economist, behavioural policy, behavioural fatigue, herd immunity behaviour, herd immunity behavior, herd immunity behavioural science, herd immunity nudg, herd immunity nudge, herd immunity nudging, herd immunity nudge unit, herd immunity nudge theory, herd immunity behavioural, nudge strategy, nanny state behaviour, nanny state nudg, nudgetheory, davidhalpern, nickchater, susanmichie, richardthaler, casssunstein, dankahneman, danielkahneman, behaviouralscience, behaviourchange, behaviouralscientist, behaviouralinsight,  libertarianpaternalism, choicearchitecture, choicearchitect, behaviouralanalysis, behaviouralanalyst, behaviouralinsightsteam, nudgeunit, behaviouraleconomics, behaviouraleconomist, behaviouralpolicy, behaviouralfatigue, herdimmunity behaviouralscience, nudgestrategy, nannystate behaviour, nannystate nudg

This resulted in a dataset of 16,568 tweets, corresponding to around 0.008% of the initial dataset.


## Dataset key figures
 
Tweet counts: 	13,664 
First tweet: 	03-03-2020 
Last tweet: 	09-06-2020

Breakdown by type of tweets: 

- original tweets: 	1,384 (10%) 
- retweets: 		9,989 (73%) 
- quotes:			1,986 (15%) 
- replies: 			305 (2%)


Link to the filtered dataset:
[http://tweetsets.library.gwu.edu/dataset/dbd5c145](http://tweetsets.library.gwu.edu/dataset/dbd5c145)


## Hydrating the tweets

We used [Hydrator](https://github.com/DocNow/hydrator) to hydrate these tweets. Hydrator manages your Twitter API Rate Limits.

After hydration, our sample consisted of 12,161 tweets.