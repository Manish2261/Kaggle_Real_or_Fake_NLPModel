# Kaggle_Real_or_Fake_NLPModel
Analyzing the Tweets data w.r.to NLP Model for scrutinizing their factfulness

## Reason for Participating in these competition:
As, we all know about the NLP model which is at the heart of Data Science, I was at first sght really amazed and inquisitive to reda such a complex problem statement for a ML model.
Also, it is challenging to build such highly sparse distributed model, and then to mine it for the required data therein.

# Introduction to the Comeptition:
Twitter has become an important communication channel in times of emergency.
The ubiquitousness of smartphones enables people to announce an emergency they’re observing in real-time. Because of this, more agencies are interested in programatically monitoring Twitter (i.e. disaster relief organizations and news agencies).

But, it’s not always clear whether a person’s words are actually announcing a disaster. Take this example:

![alt text](https://storage.googleapis.com/kaggle-media/competitions/tweet_screenshot.png)

The author explicitly uses the word “ABLAZE” but means it metaphorically. This is clear to a human right away, especially with the visual aid. But it’s less clear to a machine.So, synonimously in these competition, we are supposed to build a model which will have machine distinguish such tweets with that of the real tweets indcating some fatal calls which needs to be mitigated otherwisee, could abate great problem.

Disclaimer: The dataset for this competition contains text that may be considered profane, vulgar, or offensive.

# Method for Evaluation of Model:

Submissions are evaluated using F1 between the predicted and expected answers.

F1 is calculated as follows:
F1= 2 ∗ ((precision∗recall)/(precision+recall))
where:

precision=   (TP/(TP+FP))
recall=      (TP/(TP+FN))

and:

True Positive [TP] = your prediction is 1, and the ground truth is also 1 - you predicted a positive and that's true!

False Positive [FP] = your prediction is 1, and the ground truth is 0 - you predicted a positive, and that's false.

False Negative [FN] = your prediction is 0, and the ground truth is 1 - you predicted a negative, and that's false.

